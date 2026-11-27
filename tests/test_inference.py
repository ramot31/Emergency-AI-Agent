"""Tests unitaires pour le chaînage avant, chaînage arrière et traces (Bloc 2 - Membre 2)."""
from inference import backward_chain, format_inference_markdown, forward_chain
from interface import load_scenario
from facts import state_to_facts
from kb import KnowledgeBase
from unification import Var, apply_subst


def test_forward_chaining_p1_p2_p3():
    state = load_scenario("scenarios/scenario_complet.json")
    kb = KnowledgeBase()
    kb.load_state_facts(state_to_facts(state))

    derived, trace = forward_chain(kb, include_iteration_headers=True)

    # P1 (ChestPain + LowBP + SpO2 87 + Age 72 + ShortnessOfBreath)
    assert ("Elderly", "P1") in derived
    assert ("SevereHypoxia", "P1") in derived
    assert ("HighRisk", "P1") in derived
    assert ("CriticalRisk", "P1") in derived
    assert ("RequiresUrgentEvaluation", "P1") in derived
    assert ("RequiresICU", "P1") in derived
    assert ("RequiresSpecialist", "P1", "Cardiology") in derived
    assert ("RequiresSpecialist", "P1", "Pulmonology") in derived
    assert ("Priority", "P1", 1) in derived

    # P2 (Fever modérée, constantes stables) -> pas d'USI
    assert ("ModerateRisk", "P2") in derived
    assert ("RequiresEmergencyBed", "P2") in derived
    assert ("RequiresICU", "P2") not in derived
    assert ("CriticalRisk", "P2") not in derived

    # P3 (AbdominalPain stable) -> salle d'examen + généraliste, pas d'USI
    assert ("RequiresExamRoom", "P3") in derived
    assert ("RequiresSpecialist", "P3", "General") in derived
    assert ("RequiresICU", "P3") not in derived
    assert ("HighRisk", "P3") not in derived

    assert len(trace) > 0
    expl = kb.explain(("RequiresICU", "P1"))
    assert any("R15" in line for line in expl)

    md = format_inference_markdown(kb)
    assert "# Trace d'inférence" in md
    assert "RequiresICU(P1)" in md


def test_backward_chaining_query():
    state = load_scenario("scenarios/scenario_complet.json")
    facts = state_to_facts(state)

    proved, proofs = backward_chain(("RequiresICU", "P1"), facts)
    assert proved is True
    assert len(proofs) >= 1
    assert any("R15" in step for _, tr in proofs for step in tr)

    # Requête avec variable : qui requiert l'USI ?
    x = Var("x")
    proved_var, var_proofs = backward_chain(("RequiresICU", x), facts)
    assert proved_var is True
    matched_patients = {apply_subst(x, sub) for sub, _ in var_proofs}
    assert matched_patients == {"P1"}

    # P3 ne requiert pas l'USI
    proved_p3, _ = backward_chain(("RequiresICU", "P3"), facts)
    assert proved_p3 is False
