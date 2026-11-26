"""Démo de bout en bout + scénario dynamique : python demo_agent.py"""
import json
from interface import load_scenario
from agent import Agent

state = load_scenario("scenarios/scenario_complet.json")
agent = Agent(state)

print("=" * 20, "SCENARIO COMPLET", "=" * 20)
agent.cycle()
agent.execute_next(); agent.execute_next()           # 2 premières actions exécutées

print("\n" + "=" * 20, "CHANGEMENT : l'USI devient indisponible", "=" * 20)
agent.handle_event({"type": "resource_unavailable", "id": "ICU1"})
agent.run_plan()                                      # l'agent détecte et replanifie
print("\nETAT FINAL :", state.summary())
