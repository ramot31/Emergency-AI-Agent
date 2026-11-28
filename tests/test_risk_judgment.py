"""Politique de précaution et traçabilité règles/ML dans la simulation."""
import pytest

from kb import KnowledgeBase
from requirements_export import export_search_requirements


def evaluate(extra, level, score):
    kb = KnowledgeBase()
    kb.load_state_facts([("Patient", "P"), ("PatientWaiting", "P"), *extra,
                         ("ML_Risk", "P", level, score)])
    _, trace = kb.recompute()
    return kb, export_search_requirements(kb)["P"], " ".join(trace)


def test_rules_critical_are_not_lowered_by_ml_low():
    kb, req, trace = evaluate([("LowBloodPressure", "P"), ("OxygenSaturation", "P", 85)], "Low", 0.95)
    assert kb.contains(("CriticalRisk", "P"))
    assert req["priority"] == 3 and req["icu"] is True
    assert "désaccord" in trace and "réévaluation recommandée" in trace


def test_higher_ml_is_added_without_calling_minimum_a_contradiction():
    kb, req, trace = evaluate([("Fever", "P")], "High", 0.9)
    assert kb.contains(("ModerateRisk", "P")) and kb.contains(("HighRisk", "P"))
    assert req["priority"] == 2
    assert "complément, pas contradiction automatique" in trace


def test_agreement_is_traced():
    _, req, trace = evaluate([("LowOxygen", "P"), ("Tachycardia", "P")], "High", 0.9)
    assert req["priority"] == 2 and "accord sur le niveau minimum" in trace


def test_low_ml_does_not_remove_moderate_rules():
    kb, req, trace = evaluate([("Fever", "P")], "Low", 0.9)
    assert kb.contains(("ModerateRisk", "P")) and req["priority"] == 1
    assert "désaccord" in trace


def test_no_rule_conclusion_is_not_treated_as_low():
    _, req, trace = evaluate([], "High", 0.9)
    assert req["priority"] == 1
    assert "absence de conclusion des règles ne signifie pas Low" in trace
    assert "Medium provisoire" in trace


def test_critical_ml_is_not_lowered_without_rule():
    kb, req, trace = evaluate([], "Critical", 0.9)
    assert kb.contains(("CriticalRisk", "P")) and req["priority"] == 3


def test_rule_change_recalculates_moderated_prediction():
    kb, req, _ = evaluate([], "High", 0.9)
    assert req["priority"] == 1
    kb.assert_fact(("Fever", "P"))
    kb.recompute()
    assert export_search_requirements(kb)["P"]["priority"] == 2
    kb.retract_fact(("Fever", "P"))
    kb.recompute()
    assert export_search_requirements(kb)["P"]["priority"] == 1


def test_below_threshold_does_not_promote_high():
    kb, req, trace = evaluate([("Fever", "P")], "High", 0.4)
    assert not kb.contains(("HighRisk", "P")) and req["priority"] == 1
    assert "score sous le seuil" in trace


def test_rule_ml_trace_reports_actual_dl_conflict_resolution():
    kb = KnowledgeBase()
    kb.load_state_facts([("Patient", "P"), ("PatientWaiting", "P"),
                         ("ML_Risk", "P", "High", 0.76), ("DL_Risk", "P", "Low", 0.9)])
    _, trace = kb.recompute()
    assert export_search_requirements(kb)["P"]["priority"] == 1
    assert any("risque effectivement retenu=aucune conclusion" in line for line in trace)


def test_prediction_update_refreshes_judgment():
    kb = KnowledgeBase()
    kb.load_state_facts([("Patient", "P"), ("PatientWaiting", "P"), ("Fever", "P"),
                         ("ML_Risk", "P", "High", 0.9)])
    kb.recompute()
    assert export_search_requirements(kb)["P"]["priority"] == 2
    result = kb.update({"type": "ml_prediction", "patient": "P", "level": "Low", "prob": 0.9})
    assert export_search_requirements(kb)["P"]["priority"] == 1
    assert any("désaccord" in line and "ML=Low" in line for line in result["trace"])


@pytest.mark.parametrize("level,score", [("High", 1.2), ("High", float("nan")), ("Unknown", 0.9), ("High", "0.9")])
def test_invalid_ml_is_ignored_by_bridge(level, score):
    kb, req, trace = evaluate([], level, score)
    assert not kb.contains(("HighRisk", "P"))
    assert req["priority"] == 1
    assert "BRIDGE_ML_INVALID" in trace


def test_chest_pain_combined_rule_still_applies_under_moderation():
    kb, req, trace = evaluate([("HasSymptom", "P", "ChestPain")], "High", 0.72)
    assert kb.contains(("RequiresUrgentEvaluation", "P"))
    assert "BRIDGE_ML_CHESTPAIN" in trace
