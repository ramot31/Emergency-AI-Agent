"""Démo Bloc 1 : python main_demo.py [scenario.json] [event.json]"""
import sys, json
from interface import load_scenario, apply_event
from facts import state_to_facts, fact_str

scn = sys.argv[1] if len(sys.argv) > 1 else "scenarios/scenario_complet.json"
s = load_scenario(scn)
print(s.summary()); print("\n--- Faits symboliques ---")
for f in state_to_facts(s): print(" ", fact_str(f))

if len(sys.argv) > 2:
    before = set(state_to_facts(s))
    apply_event(s, json.load(open(sys.argv[2], encoding="utf-8")))
    after = set(state_to_facts(s))
    print("\n--- Evénement appliqué ---"); print(s.summary())
    print("Faits ajoutés :", sorted(fact_str(f) for f in after - before))
    print("Faits retirés :", sorted(fact_str(f) for f in before - after))
