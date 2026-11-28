"""Prétraitement des séries temporelles et découpage par patient (Membre 4).

Garantit l'absence de fuite de données :
1. Le découpage train / val / test est effectué par `patient_id` (et non par pas de temps).
2. Les statistiques de normalisation (moyenne, écart-type) sont calculées uniquement sur le train.
3. Les séquences sont mises à longueur fixe (`MAX_SEQ_LEN`) par padding à droite.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.model_selection import train_test_split

from .config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_SCALER_PATH,
    LABEL_TO_INDEX,
    MAX_SEQ_LEN,
    RANDOM_STATE,
    TEMPORAL_FEATURE_COLUMNS,
    VITAL_BOUNDS,
    VITAL_COLUMNS,
)
from .data_generation import load_sequences_from_csv


def _clip_snapshot(snap: Mapping[str, Any]) -> tuple[float, float, float]:
    hr = float(np.clip(float(snap["hr"]), *VITAL_BOUNDS["hr"]))
    bp = float(np.clip(float(snap["bp"]), *VITAL_BOUNDS["bp"]))
    spo2 = float(np.clip(float(snap["spo2"]), *VITAL_BOUNDS["spo2"]))
    return hr, bp, spo2


def extract_sequence_features(history: list[Mapping[str, Any]]) -> np.ndarray:
    """Transforme une liste `[{hr, bp, spo2}, ...]` en matrice `(T, 9)`."""
    if not history:
        raise ValueError("La séquence ne peut pas être vide.")

    window = history[-MAX_SEQ_LEN:]
    raw = np.array([_clip_snapshot(s) for s in window], dtype=np.float32)  # (T, 3)

    # Variations cumulées depuis t0 et variations pas à pas depuis t-1
    delta_t0 = raw - raw[0:1]
    delta_step = np.zeros_like(raw)
    if len(raw) > 1:
        delta_step[1:] = raw[1:] - raw[:-1]

    return np.concatenate([raw, delta_t0, delta_step], axis=1).astype(np.float32)


def extract_last_step_features(sequences: list[dict[str, Any]]) -> np.ndarray:
    """Extrait uniquement les 3 constantes du dernier pas pour la baseline statique."""
    rows = []
    for item in sequences:
        last = _clip_snapshot(item["history"][-1])
        rows.append(last)
    return np.array(rows, dtype=np.float32)


def split_sequences_by_patient(
    sequences: list[dict[str, Any]],
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Découpe stratifié par patient en train / validation / test."""
    labels = [item["risk_level"] for item in sequences]
    train_val, test = train_test_split(
        sequences,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )
    val_ratio = val_size / (1.0 - test_size)
    train_val_labels = [item["risk_level"] for item in train_val]
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        stratify=train_val_labels,
        random_state=random_state,
    )
    return train, val, test


@dataclass
class SequenceScaler:
    """Normalisation z-score ajustée uniquement sur les pas de temps réels du train."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, train_sequences: list[dict[str, Any]]) -> "SequenceScaler":
        all_steps = np.concatenate(
            [extract_sequence_features(item["history"]) for item in train_sequences],
            axis=0,
        )
        mean = all_steps.mean(axis=0).astype(np.float32)
        std = all_steps.std(axis=0).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        return cls(mean=mean, std=std)

    def transform_padded(self, history: list[Mapping[str, Any]]) -> tuple[np.ndarray, int]:
        """Normalise et padde à droite jusqu'à `MAX_SEQ_LEN`."""
        feats = extract_sequence_features(history)
        normed = (feats - self.mean) / self.std
        seq_len = len(normed)
        padded = np.zeros((MAX_SEQ_LEN, len(TEMPORAL_FEATURE_COLUMNS)), dtype=np.float32)
        padded[:seq_len] = normed
        return padded, seq_len

    def save(self, path: str | Path = DEFAULT_SCALER_PATH) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "features": TEMPORAL_FEATURE_COLUMNS,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path = DEFAULT_SCALER_PATH) -> "SequenceScaler":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            mean=np.array(payload["mean"], dtype=np.float32),
            std=np.array(payload["std"], dtype=np.float32),
        )


def prepare_tensors(
    sequences: list[dict[str, Any]],
    scaler: SequenceScaler,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Construit les tableaux `(X, lengths, y, trajectories)` prêts pour PyTorch."""
    X_list, len_list, y_list, traj_list = [], [], [], []
    for item in sequences:
        padded, seq_len = scaler.transform_padded(item["history"])
        X_list.append(padded)
        len_list.append(seq_len)
        y_list.append(LABEL_TO_INDEX[item["risk_level"]])
        traj_list.append(item["trajectory_type"])
    return (
        np.stack(X_list).astype(np.float32),
        np.array(len_list, dtype=np.int64),
        np.array(y_list, dtype=np.int64),
        traj_list,
    )


if __name__ == "__main__":
    seqs = load_sequences_from_csv(DEFAULT_DATASET_PATH)
    train_s, val_s, test_s = split_sequences_by_patient(seqs)
    scaler = SequenceScaler.fit(train_s)
    X_tr, l_tr, y_tr, _ = prepare_tensors(train_s, scaler)
    X_va, l_va, y_va, _ = prepare_tensors(val_s, scaler)
    X_te, l_te, y_te, _ = prepare_tensors(test_s, scaler)

    train_ids = {s["patient_id"] for s in train_s}
    val_ids = {s["patient_id"] for s in val_s}
    test_ids = {s["patient_id"] for s in test_s}
    assert not (train_ids & val_ids) and not (train_ids & test_ids) and not (val_ids & test_ids)

    print(f"Découpage par patient OK : train={len(train_s)}, val={len(val_s)}, test={len(test_s)}")
    print(f"Formes tenseurs : X_train={X_tr.shape}, X_val={X_va.shape}, X_test={X_te.shape}")
