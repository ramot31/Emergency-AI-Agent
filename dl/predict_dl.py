"""Inférence Deep Learning, re-inférence dynamique et conversion vers `DL_Risk` (Membre 4).

Respecte le contrat de `stubs.dl_predict` et `agent.py` :
- Entrée : `history` = liste de mesures `[{"hr": int, "bp": int, "spo2": int}, ...]`.
- Sortie : `(niveau, probabilité)` ou `None` si `len(history) < 3`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import torch

from .config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_SCALER_PATH,
    MIN_SEQ_LEN,
    RISK_LEVELS,
)
from .model import TemporalRiskNet
from .preprocessing import SequenceScaler


@lru_cache(maxsize=4)
def _load_cached_artifacts(model_path_str: str, scaler_path_str: str) -> tuple[TemporalRiskNet, SequenceScaler]:
    model_path = Path(model_path_str)
    scaler_path = Path(scaler_path_str)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle DL absent : {model_path}. Lancez d'abord `python -m dl.train_dl`."
        )
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler DL absent : {scaler_path}. Lancez d'abord `python -m dl.train_dl`."
        )

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = TemporalRiskNet(
        input_size=int(checkpoint["input_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
        num_classes=int(checkpoint["num_classes"]),
        cell_type=str(checkpoint["cell_type"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    scaler = SequenceScaler.load(scaler_path)
    return model, scaler


def load_dl_model(
    model_path: str | Path | None = None,
    scaler_path: str | Path | None = None,
) -> tuple[TemporalRiskNet, SequenceScaler]:
    m_path = str(Path(model_path) if model_path else DEFAULT_MODEL_PATH)
    s_path = str(Path(scaler_path) if scaler_path else DEFAULT_SCALER_PATH)
    return _load_cached_artifacts(m_path, s_path)


def predict_deterioration(
    sequence: list[Mapping[str, Any]] | Any,
    model_path: str | Path | None = None,
    scaler_path: str | Path | None = None,
) -> tuple[str, float] | None:
    """Prédit `(niveau, probabilité)` à partir d'une séquence de mesures, ou `None` si < 3 mesures."""
    history = sequence.history if hasattr(sequence, "history") else sequence
    if not history or len(history) < MIN_SEQ_LEN:
        return None

    model, scaler = load_dl_model(model_path, scaler_path)
    padded, seq_len = scaler.transform_padded(history)
    x_tensor = torch.from_numpy(padded).unsqueeze(0)  # (1, MAX_SEQ_LEN, 9)
    len_tensor = torch.tensor([seq_len], dtype=torch.int64)

    with torch.no_grad():
        logits = model(x_tensor, len_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(probs.argmax().item())
        return RISK_LEVELS[idx], round(float(probs[idx].item()), 4)


def update_and_repredict(
    history: list[Mapping[str, Any]],
    new_measurement: Mapping[str, Any],
    model_path: str | Path | None = None,
    scaler_path: str | Path | None = None,
) -> dict[str, Any]:
    """Ajoute une nouvelle mesure à la séquence et recalcule la prédiction (Bloc 6)."""
    previous = predict_deterioration(history, model_path, scaler_path)
    updated_history = [dict(step) for step in history] + [
        {
            "hr": int(new_measurement["hr"]),
            "bp": int(new_measurement["bp"]),
            "spo2": int(new_measurement["spo2"]),
        }
    ]
    current = predict_deterioration(updated_history, model_path, scaler_path)
    return {
        "history": updated_history,
        "previous": previous,
        "prediction": current,
        "changed": previous != current,
    }


def to_dl_risk_fact(
    patient_id: str,
    prediction: tuple[str, float] | None,
) -> tuple[str, str, str, float] | None:
    """Convertit la prédiction en fait Python consommé par `agent.py` et `ml_dl_bridge.py`."""
    if prediction is None:
        return None
    level, probability = prediction
    return ("DL_Risk", str(patient_id), str(level), round(float(probability), 4))


def format_dl_risk(
    patient_id: str,
    prediction: tuple[str, float] | None,
) -> str | None:
    """Représentation lisible du fait : `DL_Risk(P1,Critical,0.91)`."""
    fact = to_dl_risk_fact(patient_id, prediction)
    if fact is None:
        return None
    _, pid, level, probability = fact
    return f"DL_Risk({pid},{level},{probability:.2f})"


# Alias conformes à la fiche Membre 4 et à Agent(..., dl=dl_predict)
predict_détérioration = predict_deterioration
dl_predict = predict_deterioration


if __name__ == "__main__":
    p1_history = [
        {"hr": 105, "bp": 110, "spo2": 95},
        {"hr": 112, "bp": 105, "spo2": 93},
        {"hr": 118, "bp": 98, "spo2": 91},
        {"hr": 125, "bp": 90, "spo2": 89},
        {"hr": 132, "bp": 84, "spo2": 87},
    ]
    p1_pred = dl_predict(p1_history)
    print("P1 :", p1_pred, "->", format_dl_risk("P1", p1_pred))
    print("P2 (< 3 mesures) :", dl_predict([{"hr": 88, "bp": 120, "spo2": 97}]))

    # Simulation de re-inférence dynamique sur un patient qui se dégrade
    stable_start = [
        {"hr": 98, "bp": 112, "spo2": 95},
        {"hr": 102, "bp": 106, "spo2": 94},
        {"hr": 108, "bp": 100, "spo2": 93},
    ]
    reinf = update_and_repredict(stable_start, {"hr": 126, "bp": 86, "spo2": 88})
    print("Re-inférence :", reinf["previous"], "->", reinf["prediction"], "| changé :", reinf["changed"])
