"""Génération déterministe d'un jeu de données de triage simulé.

Les règles sont volontairement simples et pédagogiques. Elles servent à créer
un banc d'essai cohérent, jamais à prendre une décision clinique réelle.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import RANDOM_STATE, SYMPTOMS


def _risk_score(row: dict[str, float], symptoms: list[str]) -> float:
    """Score latent explicite servant uniquement à étiqueter la simulation."""
    score = 0.0
    score += 0.5 if row["age"] >= 65 else 0.0
    score += 1.5 if row["heart_rate"] >= 130 else 1.0 if row["heart_rate"] >= 110 else 0.0
    score += 2.0 if row["systolic_bp"] <= 80 else 1.0 if row["systolic_bp"] < 90 else 0.5 if row["systolic_bp"] < 100 else 0.0
    score += 2.0 if row["spo2"] <= 86 else 1.0 if row["spo2"] < 92 else 0.5 if row["spo2"] < 95 else 0.0
    score += 1.0 if row["temperature"] >= 40 else 0.5 if row["temperature"] >= 38.5 else 0.0
    score += 1.0 if row["resp_rate"] >= 30 else 0.5 if row["resp_rate"] >= 24 else 0.0
    weights = {"ChestPain": 0.75, "ShortnessOfBreath": 0.75, "Confusion": 1.25,
               "Fever": 0.25, "AbdominalPain": 0.25, "Headache": 0.0,
               "Nausea": 0.0, "Weakness": 0.25}
    return score + sum(weights[symptom] for symptom in symptoms)


def _label(score: float, row: dict[str, float]) -> str:
    if score < 1.5:
        return "Low"
    if score < 3.5:
        return "Medium"
    # Dans cette simulation, Critical exige à la fois un score global très
    # élevé et au moins une constante extrêmement dégradée. Ainsi P1 reste un
    # cas High sévère pour laisser au DL temporel le rôle de signaler une
    # détérioration critique dans le scénario commun.
    extreme_marker = row["spo2"] <= 86 or row["systolic_bp"] <= 80 or row["resp_rate"] >= 30
    if score < 6.7 or not extreme_marker:
        return "High"
    return "Critical"


def generate_dataset(output_path: str | Path, n_samples: int = 2400, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Crée un CSV synthétique avec bruit, valeurs manquantes et rares aberrations.

    Les étiquettes sont calculées avant introduction de données manquantes ou
    aberrantes. Cela permet d'évaluer le pipeline de nettoyage sans fuite.
    """
    if n_samples < 400:
        raise ValueError("n_samples doit être au moins égal à 400 pour couvrir les quatre classes.")
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []
    symptom_probabilities = np.array([0.18, 0.16, 0.06, 0.15, 0.10, 0.08, 0.08, 0.12])

    forced_high_profiles = max(240, n_samples // 10)
    for sample_index in range(n_samples):
        # Mélange de profils : la distribution reste imparfaite, comme un petit jeu réel.
        if sample_index < forced_high_profiles:
            # Profils graves mais non extrêmes, incluant le voisinage de P1.
            # Ils garantissent que High ne soit pas absorbée par Critical.
            values = {
                "age": int(np.clip(rng.normal(72, 8), 55, 90)),
                "heart_rate": int(np.clip(rng.normal(128, 8), 110, 145)),
                "systolic_bp": int(np.clip(rng.normal(85, 2.5), 81, 89)),
                "spo2": int(np.clip(rng.normal(89, 1.2), 87, 92)),
                "temperature": round(float(np.clip(rng.normal(39.0, 0.4), 38.2, 39.8)), 1),
                "resp_rate": int(np.clip(rng.normal(26, 1.8), 23, 29)),
            }
            symptoms = ["ChestPain", "ShortnessOfBreath"]
            if rng.random() < 0.18:
                symptoms.append("Weakness")
        else:
            severity = rng.choice([0, 1, 2, 3], p=[0.40, 0.29, 0.21, 0.10])
            values = {
                "age": int(np.clip(rng.normal(42 + severity * 11, 18), 18, 95)),
                "heart_rate": int(np.clip(rng.normal(76 + severity * 19, 13), 45, 185)),
                "systolic_bp": int(np.clip(rng.normal(124 - severity * 15, 17), 55, 210)),
                "spo2": int(np.clip(rng.normal(98 - severity * 3.7, 2.2), 70, 100)),
                "temperature": round(float(np.clip(rng.normal(36.8 + severity * 0.62, 0.7), 34.0, 42.0)), 1),
                "resp_rate": int(np.clip(rng.normal(16 + severity * 4.2, 3.5), 8, 42)),
            }
            probabilities = np.clip(symptom_probabilities * (1 + severity * 0.34), 0, 0.75)
            symptoms = [symptom for symptom, p in zip(SYMPTOMS, probabilities) if rng.random() < p]
        latent_score = _risk_score(values, symptoms) + float(rng.normal(0, 0.32))
        rows.append({**values, "symptoms": "|".join(symptoms), "risk_level": _label(latent_score, values)})

    frame = pd.DataFrame(rows)
    # 1.5 % de valeurs manquantes indépendantes par variable et 0.5 % d'aberrations.
    for column in ["age", "heart_rate", "systolic_bp", "spo2", "temperature", "resp_rate"]:
        frame.loc[rng.random(n_samples) < 0.015, column] = np.nan
    outlier_rows = frame.index[rng.random(n_samples) < 0.005]
    for index in outlier_rows:
        column = rng.choice(["heart_rate", "systolic_bp", "spo2", "temperature", "resp_rate"])
        frame.loc[index, column] = {"heart_rate": 320, "systolic_bp": 15, "spo2": 150,
                                    "temperature": 50, "resp_rate": 90}[column]

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return frame


if __name__ == "__main__":
    from .config import DEFAULT_DATASET_PATH

    generated = generate_dataset(DEFAULT_DATASET_PATH)
    print(f"Dataset créé : {DEFAULT_DATASET_PATH} ({len(generated)} lignes)")
