"""Explique le chemin d'un arbre ; les seuils concernent les variables prétraitées."""
from __future__ import annotations

import json

from sklearn.tree import DecisionTreeClassifier

from .config import NUMERIC_COLUMNS, SYMPTOM_COLUMNS
from .predict_ml import load_model
from .preprocessing import make_feature_frame


def explain_prediction(patient: object) -> dict:
    model = load_model()
    classifier = model.named_steps["classifier"]
    if not isinstance(classifier, DecisionTreeClassifier):
        raise ValueError("Explication de chemin disponible uniquement pour un arbre de décision.")
    features = make_feature_frame(patient)
    preprocessor = model.named_steps["preprocessor"]
    transformed = preprocessor.transform(features)
    names = NUMERIC_COLUMNS + SYMPTOM_COLUMNS
    tree = classifier.tree_
    path = classifier.decision_path(transformed).indices
    steps = []
    for node in path:
        if tree.feature[node] < 0:
            continue
        index = tree.feature[node]
        value = float(transformed[0, index])
        threshold = float(tree.threshold[node])
        step = {"variable": str(names[index]), "valeur_pretraitee": value,
                "condition": "<=" if value <= threshold else ">", "seuil_pretraite": threshold}
        if index < len(NUMERIC_COLUMNS):
            scaler = preprocessor.named_transformers_["numeric"].named_steps["scale"]
            step["seuil_unite_origine"] = float(threshold * scaler.scale_[index] + scaler.mean_[index])
        steps.append(step)
    proba = model.predict_proba(features)[0]
    winner = int(proba.argmax())
    return {"prediction": str(model.classes_[winner]), "score": float(proba[winner]),
            "probabilites": {str(c): float(p) for c, p in zip(model.classes_, proba)},
            "chemin": steps,
            "limite": "Les seuils d'origine sont approximatifs et appliqués après bornage et imputation ; les scores ne sont pas des probabilités cliniques calibrées."}


if __name__ == "__main__":
    from interface import load_scenario

    state = load_scenario("scenarios/scenario_complet.json")
    for pid in ("P1", "P2", "P3"):
        print(pid, json.dumps(explain_prediction(state.patients[pid]), ensure_ascii=False, indent=2))
