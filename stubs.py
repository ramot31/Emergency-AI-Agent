"""MODULES FACTICES (stubs). Ils permettent de faire tourner l'agent maintenant.
Chaque membre remplacera son stub par son vrai module :
  - ml_predict      -> Membre 3 (Machine Learning)
  - dl_predict      -> Membre 4 (Deep Learning)
  - reason          -> Membre 2 (moteur logique)
  - search_allocation -> Membre 5 (UCS / A*)
  - make_plan       -> Membre 6 (planificateur)
Les signatures (entrées/sorties) ci-dessous sont le CONTRAT à respecter."""

LEVEL_RANK = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}


def ml_predict(patient):
    """Membre 3. Entrée : Patient. Sortie : (niveau, probabilité)."""
    score = 0
    score += 2 if patient.spo2 < 92 else 0
    score += 2 if patient.systolic_bp < 90 else 0
    score += 1 if patient.heart_rate > 100 else 0
    score += 1 if patient.temperature >= 38 else 0
    score += 1 if patient.age > 65 else 0
    if score >= 5: return "High", 0.87
    if score >= 2: return "Medium", 0.60
    return "Low", 0.80


def dl_predict(history):
    """Membre 4. Entrée : liste de mesures [{hr,bp,spo2}, ...]. Sortie : (niveau, proba) ou None."""
    if len(history) < 3:
        return None
    first, last = history[0], history[-1]
    if last["spo2"] <= first["spo2"] - 5 and last["bp"] <= first["bp"] - 15:
        return "Critical", 0.91
    return "Low", 0.70


def reason(facts):
    """Membre 2. Entrée : liste de faits (tuples). Sortie : (faits dérivés, trace).
    Ce stub applique quelques règles à la main ; le vrai moteur fera de l'unification/résolution."""
    F = set(facts); derived, trace = set(), []
    pats = [f[1] for f in facts if f[0] == "Patient"]
    for p in pats:
        if ("HasSymptom", p, "ChestPain") in F and ("LowBloodPressure", p) in F:
            derived.add(("HighRisk", p)); trace.append(f"ChestPain({p}) ^ LowBloodPressure({p}) => HighRisk({p})")
        for f in facts:
            if f[0] == "ML_Risk" and f[1] == p and f[2] == "High" and f[3] >= 0.8:
                derived.add(("HighRisk", p)); trace.append(f"{f} => HighRisk({p})")
            if f[0] == "DL_Risk" and f[1] == p and f[2] == "Critical" and f[3] >= 0.8:
                derived.add(("CriticalRisk", p)); trace.append(f"{f} => CriticalRisk({p})")
        if ("CriticalRisk", p) in derived:
            derived.add(("HighRisk", p))
            derived.add(("RequiresICU", p)); trace.append(f"CriticalRisk({p}) => RequiresICU({p})")
        if ("HighRisk", p) in derived:
            derived.add(("RequiresUrgentEvaluation", p)); trace.append(f"HighRisk({p}) => RequiresUrgentEvaluation({p})")
        if ("HasSymptom", p, "ChestPain") in F and ("HighRisk", p) in derived:
            derived.add(("RequiresSpecialist", p, "Cardiology")); trace.append(f"HighRisk({p}) ^ ChestPain({p}) => RequiresSpecialist({p},Cardiology)")
    return derived, trace


def search_allocation(state, requirements):
    """Membre 5. Entrée : état + exigences {pid: {priority, icu, specialty}}.
    Sortie : (allocation {pid: {doctor, bed}}, patients non satisfaits).
    Ce stub est GLOUTON ; le vrai module fera UCS / A*."""
    order = sorted(requirements, key=lambda p: -requirements[p]["priority"])
    used, alloc, unsat = set(), {}, []
    for pid in order:
        req = requirements[pid]
        doc = next((d for d in state.available("doctor", specialty=req["specialty"]) if d.id not in used), None) if req["specialty"] else None
        if doc is None:
            doc = next((d for d in state.available("doctor") if d.id not in used), None)
        bed = None
        if req["icu"]:
            bed = next((b for b in state.available("bed", type="ICU") if b.id not in used), None)
            if bed is None: unsat.append((pid, "ICU"))
        if bed is None:
            bed = next((b for b in state.available("bed", type="Emergency") if b.id not in used), None)
        if doc is None or bed is None:
            unsat.append((pid, "doctor/bed")); continue
        used |= {doc.id, bed.id}
        alloc[pid] = {"doctor": doc.id, "bed": bed.id, "icu": bed.type == "ICU", "urgent": req["priority"] >= 2}
    return alloc, unsat


def make_plan(state, allocation):
    """Membre 6. Entrée : état + allocation. Sortie : liste ordonnée d'actions (tuples).
    Ce stub produit toujours la même séquence ; le vrai planificateur vérifiera préconditions/effets."""
    plan = []
    for pid, a in allocation.items():
        plan.append(("AssignDoctor", pid, a["doctor"]))
        plan.append(("ReserveICUBed" if a["icu"] else "ReserveBed", pid, a["bed"]))
        if a["urgent"]: plan.append(("UrgentEvaluation", pid))
        plan.append(("TransferPatient", pid, a["bed"]))
        plan.append(("StartMonitoring", pid))
    return plan
