"""Tests unitaires pour le pont ML/DL -> KB et la résolution de conflits (Bloc 2 - Membre 2)."""
from inference import forward_chain
from kb import KnowledgeBase
from ml_dl_bridge import BridgeThresholds, apply_ml_dl_bridge


def test_bridge_dl_overrides_ml_above_threshold():
    kb = KnowledgeBase()
    kb.assert_fact(("Patient", "P1"))
    kb.assert_fact(("HasSymptom", "P1", "ChestPain"))
    kb.assert_fact(("ML_Risk", "P1", "High", 0.87), source="ml")
    kb.assert_fact(("DL_Risk", "P1", "Critical", 0.91), source="dl")

    derived, trace = apply_ml_dl_bridge(kb)
    assert ("ResourceConflictMLDL", "P1") in derived
    assert ("CriticalRisk", "P1") in derived
    assert ("RequiresUrgentEvaluation", "P1") in derived
    assert any("DL prime" in line for line in trace)

    # Puis chaînage avant -> RequiresICU(P1) et RequiresSpecialist(P1, Cardiology)
    all_derived, _ = forward_chain(kb)
    assert ("RequiresICU", "P1") in all_derived
    assert ("RequiresSpecialist", "P1", "Cardiology") in all_derived


def test_bridge_precaution_when_dl_below_threshold():
    kb = KnowledgeBase()
    kb.assert_fact(("Patient", "P1"))
    kb.assert_fact(("ML_Risk", "P1", "High", 0.82), source="ml")
    # DL incertain (< 0.80) prédit Medium
    kb.assert_fact(("DL_Risk", "P1", "Medium", 0.55), source="dl")

    derived, trace = apply_ml_dl_bridge(kb)
    # Principe de précaution : on retient le niveau le plus élevé (High)
    assert ("HighRisk", "P1") in derived
    assert ("CriticalRisk", "P1") not in derived
    assert any("précaution" in line for line in trace)


def test_bridge_ml_high_with_chest_pain_triggers_urgent_even_without_low_bp():
    kb = KnowledgeBase()
    kb.assert_fact(("Patient", "P4"))
    kb.assert_fact(("HasSymptom", "P4", "ChestPain"))
    kb.assert_fact(("ML_Risk", "P4", "High", 0.72), source="ml")

    derived, _ = apply_ml_dl_bridge(kb, BridgeThresholds(ml_high_threshold=0.75, ml_chest_pain_threshold=0.70))
    assert ("RequiresUrgentEvaluation", "P4") in derived


def test_bridge_ignores_low_confidence_predictions():
    kb = KnowledgeBase()
    kb.assert_fact(("Patient", "P5"))
    kb.assert_fact(("DL_Risk", "P5", "Critical", 0.45), source="dl")
    derived, _ = apply_ml_dl_bridge(kb)
    assert ("CriticalRisk", "P5") not in derived
