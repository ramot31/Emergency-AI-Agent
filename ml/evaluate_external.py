"""Évaluation descriptive du modèle figé sur un jeu synthétique séparé.

Le générateur et ses hypothèses restent identiques : ce n'est pas une validation
externe clinique ni une garantie de généralisation hors de cette simulation.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .config import DEFAULT_ARTIFACT_DIR, DEFAULT_DATASET_PATH, PACKAGE_DIR, RISK_LEVELS
from .data_generation import generate_dataset
from .predict_ml import load_model
from .preprocessing import make_feature_frame

EVALUATION_SEED = 20261129
EVALUATION_PATH = PACKAGE_DIR / "data" / "independent_synthetic_evaluation.csv"


def evaluate() -> dict:
    # Fichier déjà généré : on le lit sans l'écraser. Génération reproductible sinon.
    if not EVALUATION_PATH.exists():
        generate_dataset(EVALUATION_PATH, n_samples=1200, random_state=EVALUATION_SEED)
    data = pd.read_csv(EVALUATION_PATH)
    train_data = pd.read_csv(DEFAULT_DATASET_PATH)
    columns = list(data.columns)
    overlapping = len(data.merge(train_data[columns].drop_duplicates(), on=columns, how="inner"))
    model = load_model()
    predicted = model.predict(make_feature_frame(data))
    errors = data.loc[(data["risk_level"] == "Critical") & (predicted != "Critical")].copy()
    errors["predicted_risk"] = predicted[(data["risk_level"] == "Critical") & (predicted != "Critical")]
    errors.to_csv(DEFAULT_ARTIFACT_DIR / "independent_critical_errors.csv", index=False)
    result = {"source": str(EVALUATION_PATH), "seed": EVALUATION_SEED,
              "generator": "mêmes règles que l'entraînement ; proportions différentes dues au minimum de 240 profils proches de P1",
              "warning": "Jeu synthétique distinct, non clinique et issu du même générateur ; non représentatif d'une population réelle.",
              "n_samples": len(data), "exact_rows_in_development_data": overlapping,
              "accuracy": accuracy_score(data["risk_level"], predicted),
              "classification_report": classification_report(data["risk_level"], predicted, labels=RISK_LEVELS,
                                                               output_dict=True, zero_division=0),
              "confusion_matrix_labels": RISK_LEVELS,
              "confusion_matrix": confusion_matrix(data["risk_level"], predicted, labels=RISK_LEVELS).tolist(),
              "critical_errors_by_predicted_class": errors["predicted_risk"].value_counts().to_dict()}
    destination = DEFAULT_ARTIFACT_DIR / "independent_evaluation.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
