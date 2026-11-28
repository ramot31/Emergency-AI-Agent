"""Constantes partagées pour le module Deep Learning (Membre 4)."""
from __future__ import annotations

from pathlib import Path

RANDOM_STATE = 42
RISK_LEVELS = ["Low", "Medium", "High", "Critical"]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(RISK_LEVELS)}

# Variables temporelles alignées strictement sur Patient.snapshot() (environment.py)
VITAL_COLUMNS = ["hr", "bp", "spo2"]

# Features par pas de temps : 3 constantes brutes + 3 variations cumulées depuis t0 + 3 deltas depuis t-1
TEMPORAL_FEATURE_COLUMNS = [
    "hr",
    "bp",
    "spo2",
    "delta_hr_t0",
    "delta_bp_t0",
    "delta_spo2_t0",
    "delta_hr_step",
    "delta_bp_step",
    "delta_spo2_step",
]

TRAJECTORY_TYPES = [
    "stable",
    "amelioration",
    "deterioration_lente",
    "deterioration_rapide",
]

MIN_SEQ_LEN = 3
MAX_SEQ_LEN = 8

VITAL_BOUNDS = {
    "hr": (45, 190),
    "bp": (55, 210),
    "spo2": (70, 100),
}

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = PACKAGE_DIR / "data" / "synthetic_sequences.csv"
DEFAULT_ARTIFACT_DIR = PACKAGE_DIR / "artifacts"
DEFAULT_MODEL_PATH = DEFAULT_ARTIFACT_DIR / "lstm_risk.pt"
DEFAULT_SCALER_PATH = DEFAULT_ARTIFACT_DIR / "scaler_stats.json"
DEFAULT_BASELINE_PATH = DEFAULT_ARTIFACT_DIR / "baseline_last_step.joblib"
