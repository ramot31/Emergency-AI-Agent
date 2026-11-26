"""Interface structurée (JSON) : charge un scénario ou un événement et met à jour l'état."""
import json
from environment import *


def load_scenario(path) -> State:
    data = json.load(open(path, encoding="utf-8"))
    s = State()
    for d in data.get("doctors", []): s.add_resource("doctor", Doctor(**d))
    for n in data.get("nurses", []): s.add_resource("nurse", Nurse(**n))
    for b in data.get("beds", []): s.add_resource("bed", Bed(**b))
    for r in data.get("rooms", []): s.add_resource("room", Room(**r))
    for e in data.get("equipment", []): s.add_resource("equipment", Equipment(**e))
    for p in data.get("patients", []):
        hist = p.pop("history", [])
        pat = Patient(**p); pat.history = hist
        s.add_patient(pat)
    return s


def apply_event(s: State, ev: dict):
    """Evénements : new_patient, new_measurement, resource_unavailable, resource_available."""
    t = ev["type"]
    if t == "new_patient":
        p = dict(ev["patient"]); hist = p.pop("history", [])
        pat = Patient(**p); pat.history = hist; s.add_patient(pat)
    elif t == "new_measurement":
        s.new_measurement(ev["patient"], ev["hr"], ev["bp"], ev["spo2"])
    elif t == "resource_unavailable":
        s.set_available(ev["id"], False)
    elif t == "resource_available":
        s.set_available(ev["id"], True)
    else:
        raise ValueError(f"Evénement inconnu : {t}")
    return s
