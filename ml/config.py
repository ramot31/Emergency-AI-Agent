"""Constantes partagées par le générateur, l'entraînement et l'inférence."""
from __future__ import annotations

from pathlib import Path

RANDOM_STATE = 42
RISK_LEVELS = ["Low", "Medium", "High", "Critical"]
SYMPTOMS = [
    "ChestPain",
    "ShortnessOfBreath",
    "Confusion",
    "Fever",
    "AbdominalPain",
    "Headache",
    "Nausea",
    "Weakness",
]
NUMERIC_COLUMNS = [
    "age",
    "heart_rate",
    "systolic_bp",
    "spo2",
    "temperature",
    "resp_rate",
]
SYMPTOM_COLUMNS = [f"symptom_{symptom}" for symptom in SYMPTOMS]
FEATURE_COLUMNS = NUMERIC_COLUMNS + SYMPTOM_COLUMNS

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = PACKAGE_DIR / "data" / "synthetic_triage.csv"
DEFAULT_ARTIFACT_DIR = PACKAGE_DIR / "artifacts"
DEFAULT_MODEL_PATH = DEFAULT_ARTIFACT_DIR / "risk_pipeline.joblib"
