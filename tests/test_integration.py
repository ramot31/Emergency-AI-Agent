"""Tests d'intégration : export Membre 5 et orchestrateur Agent du Membre 1 (Bloc 2 - Membre 2)."""
from agent import Agent
from facts import state_to_facts
from interface import load_scenario
from kb import KnowledgeBase, reason
from requirements_export import export_requirements, export_search_requirements


def test_export_requirements_contract_and_adapter():
    state = load_scenario("scenarios/scenario_complet.json")
    facts = state_to_facts(state) + [
        ("ML_Risk", "P1", "High", 0.87),
        ("DL_Risk", "P1", "Critical", 0.91),
        ("ML_Risk", "P2", "Low", 0.80),
        ("ML_Risk", "P3", "Low", 0.80),
    ]
    kb = KnowledgeBase()
    kb.load_state_facts(facts)
    kb.recompute()

    req = export_requirements(kb)
    assert req["P1"]["priorité"] == 1
    assert req["P1"]["usi"] == "oui"
    assert req["P1"]["spécialité"] == "Cardiology"
    assert req["P1"]["type_lit"] == "ICU"
    assert req["P1"]["salle_examen"] == "oui"

    assert req["P2"]["usi"] == "non"
    assert req["P2"]["type_lit"] == "Emergency"
    assert req["P2"]["priorité"] == 3

    assert req["P3"]["usi"] == "non"
    assert req["P3"]["spécialité"] == "General"
    assert req["P3"]["priorité"] == 4

    search_req = export_search_requirements(kb)
    assert search_req["P1"]["priority"] == 3
    assert search_req["P1"]["icu"] is True
    assert search_req["P1"]["specialty"] == "Cardiology"


def test_agent_orchestrator_runs_with_real_reasoner_and_replans():
    state = load_scenario("scenarios/scenario_complet.json")
    agent = Agent(state, reasoner=reason)

    agent.cycle()
    # P1 doit recevoir le lit ICU1 et un médecin (note : agent.py choisit une spécialité sur set)
    assert any("RequiresICU(P1)" in line for line in agent.journal)
    assert any("RequiresSpecialist(P1,Cardiology)" in line for line in agent.journal)
    agent.execute_next()
    agent.execute_next()
    assert state.assignments["P1"]["bed"] == "ICU1"
    assert state.assignments["P1"]["doctor"] in ("D1", "D2")

    # Evénement dynamique : ICU1 indisponible -> replanification
    agent.handle_event({"type": "resource_unavailable", "id": "ICU1"})
    assert agent.plan_is_obsolete()
    agent.run_plan()
    assert not state.beds["ICU1"].available
    assert state.assignments["P1"]["bed"] != "ICU1"
