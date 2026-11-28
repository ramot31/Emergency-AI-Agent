"""Constantes partagées pour le module Deep Learning (Membre 4)."""
from __future__ import annotations

from pathlib import Path

RANDOM_STATE = 42
RISK_LEVELS = ["Low", "Medium", "High", "Critical"]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(RISK_LEVELS)}

# Variables temporelles alignées strictement sur Patient.snapshot() (environment.py)
# - hr   : fréquence cardiaque / Heart Rate (battements par minute, bpm)
# - bp   : pression artérielle systolique / Blood Pressure (mmHg)
# - spo2 : saturation en oxygène dans le sang / SpO2 (%)
VITAL_COLUMNS = ["hr", "bp", "spo2"]

# Features calculées à chaque pas de temps t :
# - hr, bp, spo2                : valeurs brutes à l'instant t
# - delta_*_t0                  : variation cumulée depuis la 1re mesure (valeur(t) - valeur(t0))
# - delta_*_step                : variation entre deux mesures consécutives (valeur(t) - valeur(t-1))
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

# Bornes physiologiques réalistes (min, max) utilisées pour éviter des valeurs absurdes lors de la simulation
VITAL_BOUNDS = {
    "hr": (45, 190),     # bpm : entre 45 (bradycardie) et 190 (tachycardie sévère)
    "bp": (55, 210),     # mmHg : entre 55 (hypotension sévère) et 210 (hypertension sévère)
    "spo2": (70, 100),   # % : entre 70 % (hypoxie grave) et 100 % (normal)
}

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = PACKAGE_DIR / "data" / "synthetic_sequences.csv"
DEFAULT_ARTIFACT_DIR = PACKAGE_DIR / "artifacts"
DEFAULT_MODEL_PATH = DEFAULT_ARTIFACT_DIR / "lstm_risk.pt"
DEFAULT_SCALER_PATH = DEFAULT_ARTIFACT_DIR / "scaler_stats.json"
DEFAULT_BASELINE_PATH = DEFAULT_ARTIFACT_DIR / "baseline_last_step.joblib"
