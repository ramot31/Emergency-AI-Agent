"""Compare le vrai pont selon deux politiques ML, sans appeler le DL."""
from __future__ import annotations

import json
import argparse
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from environment import Patient, State
from facts import state_to_facts
from inference import forward_chain
from kb import KnowledgeBase
from ml.config import DEFAULT_ARTIFACT_DIR
from ml.evaluate_external import EVALUATION_PATH
from ml.predict_ml import load_model
from ml.preprocessing import make_feature_frame
from risk_judgment import _highest_risk


def compare(dataset: Path = EVALUATION_PATH, output: Path | None = None) -> dict:
    data = pd.read_csv(dataset)
    required = ["age", "heart_rate", "systolic_bp", "spo2", "temperature", "resp_rate"]
    valid = data.dropna(subset=required).copy()
    model = load_model()
    proba = model.predict_proba(make_feature_frame(valid))
    class_indices = proba.argmax(axis=1)
    rows = []
    for index, (_, row) in enumerate(valid.iterrows()):
        symptoms = [] if pd.isna(row.symptoms) else str(row.symptoms).split("|")
        patient = Patient("P", int(row.age), int(row.heart_rate), int(row.systolic_bp),
                          int(row.spo2), float(row.temperature), int(row.resp_rate), symptoms)
        state = State()
        state.add_patient(patient)
        facts = state_to_facts(state)
        kb = KnowledgeBase()
        kb.load_state_facts(facts)
        forward_chain(kb)
        rule_level = _highest_risk(kb.facts, "P")
        j = int(class_indices[index])
        level, score = str(model.classes_[j]), float(proba[index, j])
        fact = ("ML_Risk", "P", level, round(score, 4))
        decisions = {}
        for policy in ("legacy", "moderated"):
            case = KnowledgeBase()
            case.load_state_facts(facts + [fact])
            if policy == "legacy":
                # Même pont, mêmes seuils et règle douleur thoracique ; seule la
                # fonction de modération est neutralisée temporairement.
                with patch("ml_dl_bridge.moderate_ml_only_level", side_effect=lambda ml, rule: ml):
                    case.recompute()
            else:
                case.recompute()
            decisions[policy] = _highest_risk(case.facts, "P") or "Aucune"
            decisions[policy + "_urgent"] = case.contains(("RequiresUrgentEvaluation", "P"))
        rows.append({"reference": row.risk_level, "rules": rule_level or "Aucune", **decisions})
    evaluation = pd.DataFrame(rows)
    stats = {}
    for policy in ("rules", "legacy", "moderated"):
        prediction = evaluation[policy]
        reference = evaluation.reference
        stats[policy] = {
            "critical_total": int((reference == "Critical").sum()),
            "critical_not_marked_critical": int(((reference == "Critical") & (prediction != "Critical")).sum()),
            "noncritical_marked_critical": int(((reference != "Critical") & (prediction == "Critical")).sum()),
            "low_medium_total": int(reference.isin(["Low", "Medium"]).sum()),
            "low_medium_marked_high_or_critical": int((reference.isin(["Low", "Medium"]) & prediction.isin(["High", "Critical"])).sum()),
            "high_critical_total": int(reference.isin(["High", "Critical"]).sum()),
            "high_critical_below_high": int((reference.isin(["High", "Critical"]) & ~prediction.isin(["High", "Critical"])).sum()),
            "no_risk_conclusion": int((prediction == "Aucune").sum()),
        }
        if policy != "rules":
            stats[policy]["urgent_evaluations"] = int(evaluation[policy + "_urgent"].sum())
    result = {"dataset": str(dataset), "sample": len(valid), "excluded_missing_vitals": len(data) - len(valid),
              "limitations": "Même générateur synthétique ; compare le vrai pont et le vrai chaînage avec ML seul, sans DL ni allocation ; données vitales manquantes exclues. Absence de conclusion distincte de Low. Non clinique.",
              "policies": stats}
    destination = output or (DEFAULT_ARTIFACT_DIR / "policy_comparison.json")
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=EVALUATION_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.dataset, args.output), ensure_ascii=False, indent=2))
