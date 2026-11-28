"""Démonstration comparative ML -> KB -> exigences, sans modifier les autres blocs."""
from __future__ import annotations

import json

import pandas as pd

from environment import Patient, State
from facts import state_to_facts
from kb import KnowledgeBase
from requirements_export import export_search_requirements

from .config import DEFAULT_DATASET_PATH
from .predict_ml import load_model, to_ml_risk_fact
from .preprocessing import make_feature_frame


def compare(patient: Patient, prediction: tuple[str, float]) -> dict:
    state = State()
    state.add_patient(patient)
    base = state_to_facts(state)
    result = {"patient": vars(patient), "faits_environment": base,
              "prediction": to_ml_risk_fact(patient.id, prediction)}
    for name, facts in (("sans_ml", base), ("avec_ml", base + [result["prediction"]])):
        kb = KnowledgeBase()
        kb.load_state_facts(facts)
        derived, trace = kb.recompute()
        result[name] = {"faits_derives": sorted(derived, key=str), "trace": trace,
                        "exigences": export_search_requirements(kb)[patient.id]}
    return result


def find_demo() -> dict:
    """Cherche un exemple observable dans le dataset ; sélection illustrative, pas évaluation."""
    data = pd.read_csv(DEFAULT_DATASET_PATH).dropna()
    model = load_model()
    probabilities = model.predict_proba(make_feature_frame(data))
    for index, (_, row) in enumerate(data.iterrows()):
        p = probabilities[index]
        best = int(p.argmax())
        level, probability = str(model.classes_[best]), round(float(p[best]), 4)
        if level not in {"High", "Critical"}:
            continue
        patient = Patient(id="PX", age=int(row.age), heart_rate=int(row.heart_rate),
                          systolic_bp=int(row.systolic_bp), spo2=int(row.spo2),
                          temperature=float(row.temperature), resp_rate=int(row.resp_rate),
                          symptoms=row.symptoms.split("|"))
        result = compare(patient, (level, probability))
        if result["sans_ml"]["exigences"] != result["avec_ml"]["exigences"]:
            result["remarque"] = ("Exemple choisi dans les données de développement pour illustrer "
                                  "une différence ; il ne s'agit pas d'une évaluation indépendante.")
            return result
    raise RuntimeError("Aucun exemple avec changement d'exigence trouvé dans ce dataset.")


if __name__ == "__main__":
    print(json.dumps(find_demo(), ensure_ascii=False, indent=2))
