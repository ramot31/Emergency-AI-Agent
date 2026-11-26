"""Conversion de l'état en faits symboliques (utilisés par le Bloc 2). Un fait = tuple (Predicat, arg1, arg2...)."""
from environment import State, SPO2_LOW, HR_HIGH, TEMP_FEVER


def state_to_facts(s: State):
    facts = []
    for p in s.patients.values():
        facts += [("Patient", p.id), ("Age", p.id, p.age), ("HeartRate", p.id, p.heart_rate),
                  ("OxygenSaturation", p.id, p.spo2), ("Temperature", p.id, p.temperature),
                  ("RespiratoryRate", p.id, p.resp_rate)]
        lvl = p.bp_level()
        if lvl == "Low": facts.append(("LowBloodPressure", p.id))
        if lvl == "High": facts.append(("HighBloodPressure", p.id))
        if p.spo2 < SPO2_LOW: facts.append(("LowOxygen", p.id))
        if p.heart_rate > HR_HIGH: facts.append(("Tachycardia", p.id))
        if p.temperature >= TEMP_FEVER: facts.append(("Fever", p.id))
        facts += [("HasSymptom", p.id, sym) for sym in p.symptoms]
        facts.append(("PatientWaiting" if p.status == "waiting" else "UnderTreatment", p.id))
    for d in s.doctors.values():
        facts += [("Doctor", d.id), ("Specialty", d.id, d.specialty)]
        if d.available: facts.append(("Available", d.id))
    for n in s.nurses.values():
        facts.append(("Nurse", n.id))
        if n.available: facts.append(("Available", n.id))
    for b in s.beds.values():
        facts += [("Bed", b.id), ("Type", b.id, b.type)]
        if b.available: facts.append(("Available", b.id))
    for r in s.rooms.values():
        facts.append(("Room", r.id))
        if r.available: facts.append(("Available", r.id))
    for pid, a in s.assignments.items():
        facts += [(f"{k.capitalize()}Assigned", pid, rid) for k, rid in a.items()]
    return facts


def fact_str(f):
    return f"{f[0]}({','.join(str(x) for x in f[1:])})"
