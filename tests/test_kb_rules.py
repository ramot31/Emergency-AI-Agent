"""Tests unitaires pour la KB et les règles logiques (Bloc 2 - Membre 2)."""
from kb import Justification, KnowledgeBase
from rules import DEFAULT_RULES, build_default_rules


def test_default_rules_count_and_uniqueness():
    rules = build_default_rules()
    assert len(rules) >= 20
    pure_horn = [r for r in rules if r.is_pure_horn]
    assert len(pure_horn) >= 20
    ids = [r.rule_id for r in rules]
    assert len(ids) == len(set(ids))
    assert len(DEFAULT_RULES) == len(rules)


def test_kb_assert_retract_and_tms():
    kb = KnowledgeBase()
    assert kb.assert_fact(("HasSymptom", "P1", "ChestPain"), source="environment")
    assert kb.assert_fact(("LowBloodPressure", "P1"), source="environment")
    # Normalisation de casse pour HasSymptom
    assert kb.contains(("HasSymptom", "P1", "chestpain"))

    j1 = Justification("R04", (("HasSymptom", "P1", "ChestPain"), ("LowBloodPressure", "P1")), "{x/P1}")
    j2 = Justification("R14", (("HighRisk", "P1"),), "{x/P1}")
    kb.add_derived_fact(("HighRisk", "P1"), j1)
    kb.add_derived_fact(("RequiresUrgentEvaluation", "P1"), j2)
    assert kb.contains(("RequiresUrgentEvaluation", "P1"))

    # Retrait de LowBloodPressure -> HighRisk et RequiresUrgentEvaluation invalidés en cascade
    removed = kb.retract_fact(("LowBloodPressure", "P1"))
    assert ("LowBloodPressure", "P1") in removed
    assert ("HighRisk", "P1") in removed
    assert ("RequiresUrgentEvaluation", "P1") in removed
    assert not kb.contains(("HighRisk", "P1"))
    assert not kb.contains(("RequiresUrgentEvaluation", "P1"))


def test_kb_tms_preserves_fact_with_alternative_justification():
    kb = KnowledgeBase()
    kb.assert_fact(("HasSymptom", "P1", "ChestPain"))
    kb.assert_fact(("LowBloodPressure", "P1"))
    kb.assert_fact(("LowOxygen", "P1"))
    kb.assert_fact(("Tachycardia", "P1"))

    j_chest = Justification("R04", (("HasSymptom", "P1", "ChestPain"), ("LowBloodPressure", "P1")), "{x/P1}")
    j_oxy = Justification("R05", (("LowOxygen", "P1"), ("Tachycardia", "P1")), "{x/P1}")
    kb.add_derived_fact(("HighRisk", "P1"), j_chest)
    kb.add_derived_fact(("HighRisk", "P1"), j_oxy)
    kb.add_derived_fact(("RequiresUrgentEvaluation", "P1"), Justification("R14", (("HighRisk", "P1"),), "{x/P1}"))

    kb.retract_fact(("LowBloodPressure", "P1"))
    # HighRisk reste vrai grâce à j_oxy !
    assert kb.contains(("HighRisk", "P1"))
    assert kb.contains(("RequiresUrgentEvaluation", "P1"))
