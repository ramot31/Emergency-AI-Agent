"""Préparation commune des jeux d'entraînement et des patients à prédire."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_COLUMNS, NUMERIC_COLUMNS, SYMPTOMS, SYMPTOM_COLUMNS


class VitalClipper(BaseEstimator, TransformerMixin):
    """Borne les constantes à des plages physiologiquement plausibles simulées."""

    bounds = {
        "age": (0, 110), "heart_rate": (25, 240), "systolic_bp": (40, 260),
        "spo2": (50, 100), "temperature": (30, 45), "resp_rate": (5, 60),
    }

    def fit(self, x: pd.DataFrame, y: object = None) -> "VitalClipper":
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(x).copy()
        for column in frame.columns:
            if column in self.bounds:
                frame[column] = frame[column].clip(*self.bounds[column])
        return frame


def _normalise_symptoms(value: Any) -> set[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    if isinstance(value, str):
        return {item.strip() for item in value.split("|") if item.strip()}
    return {str(item).strip() for item in value if str(item).strip()}


def expand_symptoms(frame: pd.DataFrame) -> pd.DataFrame:
    """Transforme la colonne `symptoms` en indicateurs binaires stables."""
    result = frame.copy()
    symptom_sets = result.get("symptoms", pd.Series([[]] * len(result), index=result.index)).map(_normalise_symptoms)
    for symptom, column in zip(SYMPTOMS, SYMPTOM_COLUMNS):
        result[column] = symptom_sets.map(lambda items: int(symptom in items))
    return result


def make_feature_frame(patient_or_frame: Any) -> pd.DataFrame:
    """Accepte un Patient du Bloc 1, un dictionnaire ou un DataFrame."""
    if isinstance(patient_or_frame, pd.DataFrame):
        raw = patient_or_frame.copy()
    elif isinstance(patient_or_frame, Mapping):
        raw = pd.DataFrame([dict(patient_or_frame)])
    else:
        raw = pd.DataFrame([{
            "age": getattr(patient_or_frame, "age"),
            "heart_rate": getattr(patient_or_frame, "heart_rate"),
            "systolic_bp": getattr(patient_or_frame, "systolic_bp"),
            "spo2": getattr(patient_or_frame, "spo2"),
            "temperature": getattr(patient_or_frame, "temperature"),
            "resp_rate": getattr(patient_or_frame, "resp_rate"),
            "symptoms": getattr(patient_or_frame, "symptoms", []),
        }])
    missing = [column for column in NUMERIC_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"Caractéristiques patient manquantes : {', '.join(missing)}")
    features = expand_symptoms(raw)
    return features.reindex(columns=FEATURE_COLUMNS)


def make_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("clip", VitalClipper()),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    symptom_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
        ("symptoms", symptom_pipeline, SYMPTOM_COLUMNS),
    ], remainder="drop")
