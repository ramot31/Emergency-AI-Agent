"""Tests unitaires pour la FNC et la résolution par réfutation (Bloc 2 - Membre 2)."""
from facts import state_to_facts
from interface import load_scenario
from resolution import (
    And,
    Atom,
    Exists,
    Forall,
    Iff,
    Implies,
    Not,
    resolve_query,
    to_cnf,
)
from unification import Compound, Var


def test_cnf_conversion_with_skolemization_and_quantifiers():
    x, y = Var("x"), Var("y")
    # ∀x (Patient(x) -> ∃y DoctorAssigned(x, y))
    f1 = Forall(x, Implies(Atom(("Patient", x)), Exists(y, Atom(("DoctorAssigned", x, y)))))
    clauses = to_cnf(f1, source="test_skolem")
    assert len(clauses) == 1
    cl = clauses[0]
    assert len(cl.literals) == 2
    assert cl.literals[0].negated is True
    assert cl.literals[1].negated is False
    # La variable existentielle y sous ∀x est skolémisée en sk_1(x)
    assert isinstance(cl.literals[1].atom[2], Compound)

    # Équivalence + double négation : (P <-> Q) produit 2 clauses
    f2 = Iff(Not(Not(Atom(("HighRisk", "P1")))), Atom(("RequiresUrgentEvaluation", "P1")))
    clauses2 = to_cnf(f2, source="test_iff")
    assert len(clauses2) == 2


def test_resolution_examples_success_and_failure():
    state = load_scenario("scenarios/scenario_complet.json")
    facts = state_to_facts(state) + [("DL_Risk", "P1", "Critical", 0.91)]

    # 1. Succès : RequiresICU(P1)? via règles (R01/R03/R09/R10 -> CriticalRisk -> R15 RequiresICU)
    res1 = resolve_query(("RequiresICU", "P1"), facts)
    assert res1.proved is True
    assert res1.proof_steps[-1].resolvent.is_empty
    assert "PROUVÉ" in res1.format_trace(only_proof_path=True)

    # 2. Succès : RequiresUrgentEvaluation(P1)?
    res2 = resolve_query(("RequiresUrgentEvaluation", "P1"), facts)
    assert res2.proved is True

    # 3. Succès : RequiresSpecialist(P1, Cardiology)?
    res3 = resolve_query(("RequiresSpecialist", "P1", "Cardiology"), facts)
    assert res3.proved is True

    # 4. Succès avec variable : ∃x RequiresICU(x)?
    x = Var("x")
    res4 = resolve_query(("RequiresICU", x), facts)
    assert res4.proved is True

    # 5. Non-dérivable : RequiresICU(P2)? (P2 a seulement une fièvre modérée)
    res5 = resolve_query(("RequiresICU", "P2"), facts)
    assert res5.proved is False

    # 6. Non-dérivable : RequiresICU(P3)? (P3 est stable)
    res6 = resolve_query(("RequiresICU", "P3"), facts)
    assert res6.proved is False


def test_resolution_respects_step_limit():
    state = load_scenario("scenarios/scenario_complet.json")
    facts = state_to_facts(state)
    res = resolve_query(("RequiresICU", "P3"), facts, max_steps=10, max_depth=3)
    assert res.proved is False
    assert res.explored_steps <= 10
