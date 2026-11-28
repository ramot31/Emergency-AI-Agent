"""Génération déterministe de séries temporelles de constantes vitales (Membre 4).

Simulation pédagogique uniquement (aucun usage médical réel).
Chaque patient possède une séquence de longueur variable (3 à 8 pas de temps)
au format compatible avec `Patient.history` : `[{"hr": ..., "bp": ..., "spo2": ...}, ...]`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_DATASET_PATH,
    MAX_SEQ_LEN,
    MIN_SEQ_LEN,
    RANDOM_STATE,
    TRAJECTORY_TYPES,
    VITAL_BOUNDS,
)


def _clip_vital(name: str, value: float) -> int:
    low, high = VITAL_BOUNDS[name]
    return int(round(float(np.clip(value, low, high))))


def _compute_sequence_risk(history: list[dict[str, int]], trajectory: str, noise: float) -> str:
    """Étiquette de risque combinant l'état final et la dynamique temporelle.

    Contrairement au modèle statique (dernier pas uniquement), la règle tient
    compte de la pente de dégradation (hausse de FC, baisse de PA et SpO2).
    """
    first, last = history[0], history[-1]
    delta_hr = last["hr"] - first["hr"]
    delta_bp = last["bp"] - first["bp"]
    delta_spo2 = last["spo2"] - first["spo2"]

    # Score de sévérité au dernier instant
    last_score = 0.0
    last_score += 1.6 if last["hr"] >= 130 else 0.9 if last["hr"] >= 110 else 0.3 if last["hr"] >= 98 else 0.0
    last_score += 2.2 if last["bp"] <= 82 else 1.3 if last["bp"] < 92 else 0.5 if last["bp"] < 102 else 0.0
    last_score += 2.2 if last["spo2"] <= 86 else 1.4 if last["spo2"] < 92 else 0.5 if last["spo2"] < 95 else 0.0

    # Score de tendance temporelle (détérioration vs amélioration)
    trend_score = 0.0
    if delta_hr >= 20:
        trend_score += 1.2
    elif delta_hr >= 10:
        trend_score += 0.6
    elif delta_hr <= -12:
        trend_score -= 0.6

    if delta_bp <= -20:
        trend_score += 1.4
    elif delta_bp <= -10:
        trend_score += 0.7
    elif delta_bp >= 12:
        trend_score -= 0.6

    if delta_spo2 <= -6:
        trend_score += 1.6
    elif delta_spo2 <= -3:
        trend_score += 0.8
    elif delta_spo2 >= 3:
        trend_score -= 0.6

    total = last_score + trend_score + noise

    # Garde-fou : une détérioration rapide marquée (type P1) vers des constantes
    # dégradées est critique même si une constante n'est pas encore extrême.
    rapid_shock = (
        trajectory == "deterioration_rapide"
        and delta_hr >= 18
        and delta_bp <= -15
        and delta_spo2 <= -5
    )
    if (total >= 5.2 and (last["spo2"] <= 89 or last["bp"] <= 88 or rapid_shock)) or total >= 6.4:
        return "Critical"
    if total >= 2.8:
        return "High"
    if total >= 1.1:
        return "Medium"
    return "Low"


def _generate_patient_sequence(
    rng: np.random.Generator,
    trajectory: str,
    length: int,
) -> list[dict[str, int]]:
    """Génère une trajectoire physiologique simulée sur `length` pas de temps."""
    if trajectory == "stable":
        # Mélange de patients stables sains et chroniques modérés
        is_moderate = rng.random() < 0.35
        hr0 = rng.normal(96 if is_moderate else 76, 8)
        bp0 = rng.normal(106 if is_moderate else 122, 9)
        spo2_0 = rng.normal(94.5 if is_moderate else 97.8, 1.2)
        hr_step, bp_step, spo2_step = 0.0, 0.0, 0.0
    elif trajectory == "amelioration":
        hr0 = rng.normal(114, 9)
        bp0 = rng.normal(96, 8)
        spo2_0 = rng.normal(91.5, 1.8)
        hr_step = rng.uniform(-5.5, -2.5)
        bp_step = rng.uniform(2.5, 5.5)
        spo2_step = rng.uniform(0.7, 1.5)
    elif trajectory == "deterioration_lente":
        hr0 = rng.normal(92, 9)
        bp0 = rng.normal(114, 9)
        spo2_0 = rng.normal(95.0, 1.5)
        hr_step = rng.uniform(2.0, 4.5)
        bp_step = rng.uniform(-4.5, -2.0)
        spo2_step = rng.uniform(-1.3, -0.5)
    elif trajectory == "deterioration_rapide":
        # Inclut des profils proches de P1 (105 -> 132, 110 -> 84, 95 -> 87)
        hr0 = rng.normal(104, 8)
        bp0 = rng.normal(112, 8)
        spo2_0 = rng.normal(94.8, 1.5)
        hr_step = rng.uniform(5.5, 9.5)
        bp_step = rng.uniform(-8.5, -5.0)
        spo2_step = rng.uniform(-2.4, -1.3)
    else:
        raise ValueError(f"Trajectoire inconnue : {trajectory}")

    history: list[dict[str, int]] = []
    for t in range(length):
        hr_t = _clip_vital("hr", hr0 + t * hr_step + rng.normal(0, 2.2))
        bp_t = _clip_vital("bp", bp0 + t * bp_step + rng.normal(0, 2.5))
        spo2_t = _clip_vital("spo2", spo2_0 + t * spo2_step + rng.normal(0, 0.7))
        history.append({"hr": hr_t, "bp": bp_t, "spo2": spo2_t})
    return history


def generate_sequential_dataset(
    output_path: str | Path = DEFAULT_DATASET_PATH,
    n_patients: int = 1600,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Génère un jeu de données séquentielles documenté et l'enregistre en CSV."""
    if n_patients < 200:
        raise ValueError("n_patients doit être au moins égal à 200.")

    rng = np.random.default_rng(random_state)
    traj_probs = [0.30, 0.22, 0.24, 0.24]
    rows: list[dict[str, Any]] = []

    for idx in range(n_patients):
        pid = f"DL_P{idx + 1:04d}"
        trajectory = str(rng.choice(TRAJECTORY_TYPES, p=traj_probs))
        length = int(rng.integers(MIN_SEQ_LEN, MAX_SEQ_LEN + 1))
        history = _generate_patient_sequence(rng, trajectory, length)
        label_noise = float(rng.normal(0, 0.25))
        risk_level = _compute_sequence_risk(history, trajectory, label_noise)

        for step, snap in enumerate(history, start=1):
            rows.append(
                {
                    "patient_id": pid,
                    "timestep": step,
                    "hr": snap["hr"],
                    "bp": snap["bp"],
                    "spo2": snap["spo2"],
                    "trajectory_type": trajectory,
                    "risk_level": risk_level,
                }
            )

    frame = pd.DataFrame(rows)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return frame


def load_sequences_from_csv(
    csv_path: str | Path = DEFAULT_DATASET_PATH,
) -> list[dict[str, Any]]:
    """Charge le CSV et regroupe chaque patient en `{patient_id, history, trajectory_type, risk_level}`."""
    df = pd.read_csv(csv_path)
    df = df.sort_values(["patient_id", "timestep"])
    sequences: list[dict[str, Any]] = []
    for pid, group in df.groupby("patient_id", sort=False):
        history = [
            {"hr": int(r.hr), "bp": int(r.bp), "spo2": int(r.spo2)}
            for r in group.itertuples(index=False)
        ]
        sequences.append(
            {
                "patient_id": str(pid),
                "history": history,
                "trajectory_type": str(group["trajectory_type"].iloc[0]),
                "risk_level": str(group["risk_level"].iloc[0]),
            }
        )
    return sequences


if __name__ == "__main__":
    df = generate_sequential_dataset()
    patients_df = df.groupby("patient_id").first()
    print(f"Dataset généré : {len(patients_df)} patients, {len(df)} mesures ({DEFAULT_DATASET_PATH})")
    print("Distribution des trajectoires :\n", patients_df["trajectory_type"].value_counts())
    print("Distribution des classes :\n", patients_df["risk_level"].value_counts())
