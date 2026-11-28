"""Entraîne, compare et sauvegarde le module ML reproductible du membre 3."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, make_scorer, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import DEFAULT_ARTIFACT_DIR, DEFAULT_DATASET_PATH, RANDOM_STATE, RISK_LEVELS
from .data_generation import generate_dataset
from .predict_ml import format_ml_risk
from .preprocessing import make_feature_frame, make_preprocessor


def critical_recall(y_true: Any, y_pred: Any) -> float:
    return recall_score(y_true, y_pred, labels=["Critical"], average="macro", zero_division=0)


def select_cv_index(results: dict[str, Any]) -> int:
    """Choix lexicographique effectué uniquement sur les scores de validation croisée."""
    return max(range(len(results["params"])), key=lambda index: (
        results["mean_test_critical_recall"][index], results["mean_test_f1_macro"][index]
    ))


def _models() -> dict[str, tuple[object, dict[str, list[object]]]]:
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
            {"classifier__C": [0.3, 1.0, 3.0]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {"classifier__max_depth": [5, 8, None], "classifier__min_samples_leaf": [2, 6]},
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1),
            {"classifier__n_estimators": [160], "classifier__max_depth": [8, None],
             "classifier__min_samples_leaf": [1, 3]},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"classifier__n_neighbors": [5, 11, 21], "classifier__weights": ["uniform", "distance"]},
        ),
        "SVM": (
            CalibratedClassifierCV(SVC(class_weight="balanced", random_state=RANDOM_STATE), cv=3),
            {"classifier__estimator__C": [0.3, 1.0, 3.0], "classifier__estimator__kernel": ["linear", "rbf"]},
        ),
    }


def _svg_document(width: int, height: int, body: str) -> str:
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>{body}</svg>')


def _save_confusion_matrix(name: str, y_true: pd.Series, y_pred: np.ndarray, output: Path) -> None:
    """Produit un SVG autonome afin de ne pas dépendre d'un backend graphique."""
    matrix = confusion_matrix(y_true, y_pred, labels=RISK_LEVELS)
    maximum = max(int(matrix.max()), 1)
    x0, y0, cell = 150, 92, 85
    body = [f'<text x="20" y="32" font-size="20" font-family="Arial" font-weight="bold">Matrice de confusion - {html.escape(name)}</text>',
            '<text x="290" y="70" text-anchor="middle" font-size="14" font-family="Arial">Classe prédite</text>',
            '<text x="22" y="260" transform="rotate(-90 22,260)" text-anchor="middle" font-size="14" font-family="Arial">Classe réelle</text>']
    for position, level in enumerate(RISK_LEVELS):
        x = x0 + position * cell + cell / 2
        y = y0 + position * cell + cell / 2 + 5
        body.append(f'<text x="{x}" y="82" text-anchor="middle" font-size="12" font-family="Arial">{level}</text>')
        body.append(f'<text x="140" y="{y}" text-anchor="end" font-size="12" font-family="Arial">{level}</text>')
    for row in range(4):
        for column in range(4):
            value = int(matrix[row, column])
            shade = int(245 - 155 * value / maximum)
            x, y = x0 + column * cell, y0 + row * cell
            body.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="rgb({shade},{shade + 25},{245})" stroke="white"/>')
            body.append(f'<text x="{x + cell / 2}" y="{y + cell / 2 + 5}" text-anchor="middle" font-size="14" font-family="Arial">{value}</text>')
    (output / f"confusion_{name.lower().replace(' ', '_')}.svg").write_text(
        _svg_document(540, 470, "".join(body)), encoding="utf-8"
    )


def _save_feature_importance(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series, output: Path) -> list[dict[str, float]]:
    from sklearn.inspection import permutation_importance

    result = permutation_importance(model, x_test, y_test, scoring="f1_macro", n_repeats=8,
                                    random_state=RANDOM_STATE, n_jobs=-1)
    ranking = (pd.DataFrame({"feature": x_test.columns, "importance_mean": result.importances_mean,
                             "importance_std": result.importances_std})
               .sort_values("importance_mean", ascending=False))
    ranking.to_csv(output / "feature_importance.csv", index=False)
    maximum = max(float(ranking["importance_mean"].max()), 0.001)
    body = ['<text x="20" y="30" font-size="20" font-family="Arial" font-weight="bold">Importance par permutation</text>',
            '<text x="20" y="52" font-size="13" font-family="Arial">Baisse de F1 macro après permutation</text>']
    for index, row in enumerate(ranking.itertuples(index=False)):
        y = 78 + index * 30
        width = max(0.0, float(row.importance_mean)) / maximum * 350
        body.append(f'<text x="180" y="{y + 15}" text-anchor="end" font-size="12" font-family="Arial">{html.escape(str(row.feature))}</text>')
        body.append(f'<rect x="190" y="{y}" width="{width:.1f}" height="19" fill="#2878b5"/>')
        body.append(f'<text x="{198 + width:.1f}" y="{y + 15}" font-size="11" font-family="Arial">{float(row.importance_mean):.4f}</text>')
    height = 98 + len(ranking) * 30
    (output / "feature_importance.svg").write_text(_svg_document(650, height, "".join(body)), encoding="utf-8")
    return ranking.to_dict(orient="records")


def train(dataset_path: str | Path = DEFAULT_DATASET_PATH, artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR) -> dict[str, Any]:
    """Exécute la comparaison réelle et retourne les métriques sauvegardées."""
    dataset = Path(dataset_path)
    if not dataset.exists():
        generate_dataset(dataset)
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(dataset)
    if set(RISK_LEVELS) - set(data["risk_level"].unique()):
        raise ValueError("Le dataset doit contenir les quatre classes de risque.")

    x = make_feature_frame(data)
    y = data["risk_level"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    comparisons: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}

    for name, (classifier, parameters) in _models().items():
        pipeline = Pipeline([("preprocessor", make_preprocessor()), ("classifier", classifier)])
        search = GridSearchCV(pipeline, parameters,
                              scoring={"critical_recall": make_scorer(critical_recall), "f1_macro": "f1_macro"},
                              cv=cv, n_jobs=-1, refit=select_cv_index)
        search.fit(x_train, y_train)
        comparisons.append({
            "model": name, "best_params": search.best_params_,
            "cv_critical_recall": float(search.cv_results_["mean_test_critical_recall"][search.best_index_]),
            "cv_f1_macro": float(search.cv_results_["mean_test_f1_macro"][search.best_index_]),
        })
        fitted_models[name] = search.best_estimator_

    # Le modèle est figé avant tout calcul sur le jeu de test.
    selected = max(comparisons, key=lambda result: (result["cv_critical_recall"], result["cv_f1_macro"]))
    selected_name = str(selected["model"])
    final_model = fitted_models[selected_name]
    for row in comparisons:
        name = row["model"]
        prediction = fitted_models[name].predict(x_test)
        report = classification_report(y_test, prediction, labels=RISK_LEVELS, output_dict=True, zero_division=0)
        row.update({
            "accuracy": float(accuracy_score(y_test, prediction)),
            "precision_macro": float(precision_score(y_test, prediction, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, prediction, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, prediction, average="macro", zero_division=0)),
            "critical_recall": float(critical_recall(y_test, prediction)),
            "classification_report": report,
        })
        _save_confusion_matrix(name, y_test, prediction, output)

    model_path = output / "risk_pipeline.joblib"
    joblib.dump(final_model, model_path)

    importance = _save_feature_importance(final_model, x_test, y_test, output)
    p1 = {"id": "P1", "age": 72, "heart_rate": 132, "systolic_bp": 84, "spo2": 87,
          "temperature": 39.0, "resp_rate": 26, "symptoms": ["ChestPain", "ShortnessOfBreath"]}
    probabilities = final_model.predict_proba(make_feature_frame(p1))[0]
    position = int(np.argmax(probabilities))
    p1_prediction = (str(final_model.classes_[position]), round(float(probabilities[position]), 4))
    summary = {
        "random_state": RANDOM_STATE,
        "dataset": str(dataset),
        "n_samples": int(len(data)),
        "class_distribution": {str(k): int(v) for k, v in y.value_counts().reindex(RISK_LEVELS).items()},
        "split": {"train": int(len(x_train)), "test": int(len(x_test)), "stratified": True},
        "selection_rule": "maximiser le rappel Critical en validation croisée, puis le F1 macro en validation croisée ; test après sélection",
        "selected_model": selected_name,
        "selected_model_path": str(model_path),
        "models": comparisons,
        "feature_importance": importance,
        "p1_prediction": {"prediction": list(p1_prediction), "fact": format_ml_risk("P1", p1_prediction)},
    }
    (output / "evaluation_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{key: value for key, value in row.items() if key != "classification_report"}
                  for row in comparisons]).to_csv(output / "model_comparison.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    arguments = parser.parse_args()
    results = train(arguments.dataset, arguments.artifacts)
    print(f"Modèle retenu : {results['selected_model']}")
    print(f"Prédiction P1 : {results['p1_prediction']['fact']}")
    print(f"Résultats : {arguments.artifacts / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
