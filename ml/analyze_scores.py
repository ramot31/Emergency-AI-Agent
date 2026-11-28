"""Analyse descriptive des scores et de seuils du modèle figé, sans DL."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import DEFAULT_ARTIFACT_DIR
from .evaluate_external import EVALUATION_PATH
from .predict_ml import load_model
from .preprocessing import make_feature_frame


def analyze() -> dict:
    data = pd.read_csv(EVALUATION_PATH)
    model = load_model()
    probabilities = model.predict_proba(make_feature_frame(data))
    classes = list(model.classes_)
    actual = data.risk_level.to_numpy()
    indices = probabilities.argmax(axis=1)
    prediction = np.asarray(classes)[indices]
    confidence = probabilities[np.arange(len(data)), indices]
    correct = prediction == actual
    indicators = np.column_stack([actual == label for label in classes]).astype(float)
    brier = float(np.mean(np.sum((probabilities - indicators) ** 2, axis=1)))
    groups = []
    for lower, upper in ((0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 1.0000001)):
        mask = (confidence >= lower) & (confidence < upper)
        if mask.any():
            groups.append({"score_min_inclus": lower, "score_max_exclu": min(upper, 1.0),
                           "inclut_un_si_max": upper > 1, "effectif": int(mask.sum()),
                           "score_moyen": float(confidence[mask].mean()),
                           "proportion_correcte": float(correct[mask].mean())})
    thresholds = []
    critical = actual == "Critical"
    elevated_reference = np.isin(actual, ["High", "Critical"])
    for threshold in (0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95):
        # Étude de la sortie du ML seul : une non-acceptation n'est pas une
        # classe Low et n'est pas une décision du pont ni de la KB.
        accepted = confidence >= threshold
        accepted_critical = accepted & (prediction == "Critical")
        accepted_elevated = accepted & np.isin(prediction, ["High", "Critical"])
        thresholds.append({"threshold": threshold, "accepted": int(accepted.sum()),
                           "accepted_fraction": float(accepted.mean()),
                           "critical_total": int(critical.sum()),
                           "critical_not_accepted_as_critical": int((critical & ~accepted_critical).sum()),
                           "noncritical_accepted_as_critical": int((~critical & accepted_critical).sum()),
                           "elevated_total": int(elevated_reference.sum()),
                           "elevated_not_accepted_as_elevated": int((elevated_reference & ~accepted_elevated).sum()),
                           "low_medium_accepted_as_elevated": int((~elevated_reference & accepted_elevated).sum())})
    result = {"n": len(data), "classes": classes,
              "dataset": "jeu synthétique complémentaire déjà consulté, même générateur",
              "notice": "Analyse descriptive uniquement ; pas de calibration clinique ni de sélection de seuil indépendante. Les seuils décrivent le ML seul et ne reproduisent pas les règles ou le pont.",
              "multiclass_brier_sum_of_squared_errors": brier,
              "winning_score_bins": groups, "thresholds": thresholds}
    (DEFAULT_ARTIFACT_DIR / "score_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(thresholds).to_csv(DEFAULT_ARTIFACT_DIR / "threshold_analysis.csv", index=False)
    return result


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2))
