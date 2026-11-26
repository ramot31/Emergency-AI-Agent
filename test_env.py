"""Tests simples : lancer avec  python -m pytest -q   (ou  python test_env.py)"""
from interface import load_scenario, apply_event
from facts import state_to_facts
from agent import Agent


def fresh(): return load_scenario("scenarios/scenario_complet.json")


def test_facts_p1():
    f = set(state_to_facts(fresh()))
    assert ("LowBloodPressure", "P1") in f and ("HasSymptom", "P1", "ChestPain") in f
    assert ("LowBloodPressure", "P2") not in f


def test_event_icu_indisponible():
    s = fresh(); apply_event(s, {"type": "resource_unavailable", "id": "ICU1"})
    assert ("Available", "ICU1") not in set(state_to_facts(s))


def test_copy_independante():
    s = fresh(); c = s.copy(); c.assign("P1", doctor="D1")
    assert s.doctors["D1"].available and not c.doctors["D1"].available


def test_assign_release():
    s = fresh(); s.assign("P1", doctor="D1", bed="ICU1"); s.release("P1")
    assert s.doctors["D1"].available and s.beds["ICU1"].available


def test_agent_replanifie():
    s = fresh(); a = Agent(s); a.cycle(); a.execute_next()
    a.handle_event({"type": "resource_unavailable", "id": "ICU1"})
    assert a.plan_is_obsolete()
    a.run_plan()
    assert not s.beds["ICU1"].available and s.assignments["P1"]["bed"] != "ICU1"


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"): fn(); print("OK", n)
