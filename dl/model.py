"""Architecture PyTorch LSTM / GRU pour la prédiction de risque temporel (Membre 4).

ce fichier décrit la structure du réseau de neurones ("le cerveau").
Il est séparé de l'entraînement (`train_dl.py`) pour être réutilisé à l'identique
au moment de prédire sur de nouveaux patients dans l'agent (`predict_dl.py`).

Ce qu'il fait étape par étape :
1. Les patients ont entre 3 et 8 mesures ; on complète les plus courts avec des 0
   pour qu'ils aient tous la même taille en mémoire.
2. `pack_padded_sequence` dit au réseau d'ignorer ces 0 ajoutés à la fin.
3. Le LSTM (ou GRU) lit les vraies mesures dans l'ordre (t1, t2, t3...) pour
   repérer si l'état du patient s'améliore, reste stable ou se dégrade.
4. À la fin de la séquence, il sort 4 scores (un pour Low, Medium, High, Critical).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class TemporalRiskNet(nn.Module):
    """Réseau récurrent (LSTM ou GRU) traitant des séquences de longueurs variables."""

    def __init__(
        self,
        input_size: int = 9,
        hidden_size: int = 48,
        num_layers: int = 2,
        dropout: float = 0.25,
        num_classes: int = 4,
        cell_type: str = "lstm",
    ) -> None:
        super().__init__()
        cell_type = cell_type.lower()
        if cell_type not in {"lstm", "gru"}:
            raise ValueError(f"cell_type doit être 'lstm' ou 'gru', reçu : {cell_type}")

        self.cell_type = cell_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.num_classes = num_classes

        rnn_cls = nn.LSTM if cell_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Passe avant avec `pack_padded_sequence` pour ignorer le padding."""
        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        if self.cell_type == "lstm":
            _, (h_n, _) = self.rnn(packed)
        else:
            _, h_n = self.rnn(packed)
        last_hidden = h_n[-1]  # (batch, hidden_size)
        return self.head(last_hidden)
