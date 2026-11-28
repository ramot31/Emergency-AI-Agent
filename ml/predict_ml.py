"""Inférence ML et adaptation vers le contrat de la base de connaissances."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Mapping
import math
from numbers import Real

import joblib

from .config import DEFAULT_MODEL_PATH, NUMERIC_COLUMNS, SYMPTOMS
from .preprocessing import VitalClipper, _normalise_symptoms, make_feature_frame


def load_model(model_path: str | Path | None = None):
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Modèle ML absent : {path}. Lancez d'abord `python -m ml.train_ml`."
        )
    return joblib.load(path)


def predict_risk(patient: Any, model_path: str | Path | None = None) -> tuple[str, float]:
    """Retourne `(niveau, probabilité)` pour un Patient ou un dictionnaire du Bloc 1."""
    model = load_model(model_path)
    features = make_feature_frame(patient)
    probabilities = model.predict_proba(features)[0]
    index = int(probabilities.argmax())
    return str(model.classes_[index]), round(float(probabilities[index]), 4)


def inspect_patient(patient: Any) -> list[str]:
    """Signale les limites connues des données, sans modifier le Patient."""
    get = patient.get if isinstance(patient, Mapping) else lambda key, default=None: getattr(patient, key, default)
    warnings = []
    for column in NUMERIC_COLUMNS:
        value = get(column)
        if value is None or (isinstance(value, Real) and math.isnan(value)):
            warnings.append(f"{column} manquant : imputation médiane si prédiction possible")
        elif isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            warnings.append(f"{column} invalide : fournir un nombre fini")
        elif not VitalClipper.bounds[column][0] <= value <= VitalClipper.bounds[column][1]:
            warnings.append(f"{column} hors des bornes de simulation : valeur bornée par le pipeline")
    symptoms = get("symptoms")
    if symptoms is None:
        warnings.append("symptoms absent ou inconnu : traité comme aucune présence connue")
    else:
        unknown = sorted(_normalise_symptoms(symptoms) - set(SYMPTOMS))
        if unknown:
            warnings.append("symptômes non reconnus, ignorés par le modèle : " + ", ".join(unknown))
    return warnings


def predict_risk_with_diagnostics(patient: Any, model_path: str | Path | None = None) -> dict:
    """Interface complémentaire : ne change pas le contrat `predict_risk`."""
    warnings = inspect_patient(patient)
    missing_columns = [column for column in NUMERIC_COLUMNS
                       if (column not in patient if isinstance(patient, Mapping) else not hasattr(patient, column))]
    if missing_columns:
        raise ValueError(f"Champs requis absents : {', '.join(missing_columns)}")
    level, score = predict_risk(patient, model_path)
    return {"level": level, "score": score, "warnings": warnings,
            "notice": "Simulation : score non validé comme probabilité médicale ; avertissements non transmis automatiquement à l'agent."}


def to_ml_risk_fact(patient_id: str, prediction: tuple[str, float]) -> tuple[str, str, str, float]:
    """Format tuple réellement consommé par `agent.py` et `ml_dl_bridge.py`."""
    level, probability = prediction
    return ("ML_Risk", str(patient_id), str(level), round(float(probability), 4))


def format_ml_risk(patient_id: str, prediction: tuple[str, float]) -> str:
    """Représentation lisible du fait : `ML_Risk(P1,High,0.87)`."""
    _, pid, level, probability = to_ml_risk_fact(patient_id, prediction)
    return f"ML_Risk({pid},{level},{probability:.2f})"


# Alias explicite pour injection dans Agent(..., ml=ml_predict).
ml_predict = predict_risk
