"""Export des exigences logiques vers le module de recherche (Membre 5 - Contrat n°5).

Avertissement : ce module fait partie d'un prototype pédagogique de simulation
d'aide à la décision aux urgences, sans aucun usage médical réel.

Expose :
- `export_requirements(kb)` : format exact du contrat n°5 (`priorité` 1..4,
  `usi`: "oui"/"non", `spécialité`, `type_lit`, `salle_examen`, etc.) ;
- `export_search_requirements(kb)` : adaptateur compatible avec `stubs.search_allocation`
  et `agent.py` (`priority` 3..1, `icu` bool, `specialty` str|None) ;
- `reason` : réexporté depuis `kb.py` pour remplacer `stubs.reason`.
"""
from __future__ import annotations

from typing import Any

from kb import KnowledgeBase, reason

SPECIALTY_PRECEDENCE: dict[str, int] = {
    "Cardiology": 0,
    "Neurology": 1,
    "Pulmonology": 2,
    "InfectiousDisease": 3,
    "Gastroenterology": 4,
    "General": 5,
}


def _patient_ids(kb: KnowledgeBase, only_waiting: bool) -> list[str]:
    all_facts = kb.facts
    if only_waiting:
        ids = [f[1] for f in all_facts if len(f) == 2 and f[0] == "PatientWaiting"]
    else:
        ids = [f[1] for f in all_facts if len(f) == 2 and f[0] == "Patient"]
    return sorted(set(str(pid) for pid in ids))


def _effective_priority_1_to_4(all_facts: set[tuple[Any, ...]], pid: str) -> int:
    prios = [int(f[2]) for f in all_facts if len(f) == 3 and f[0] == "Priority" and f[1] == pid]
    return min(prios) if prios else 4


def _ordered_specialties(all_facts: set[tuple[Any, ...]], pid: str) -> list[str]:
    specs = {str(f[2]) for f in all_facts if len(f) == 3 and f[0] == "RequiresSpecialist" and f[1] == pid}
    return sorted(specs, key=lambda s: (SPECIALTY_PRECEDENCE.get(s, 99), s))


def export_requirements(kb: KnowledgeBase, only_waiting: bool = True) -> dict[str, dict[str, Any]]:
    """Exporte les exigences dérivées par patient selon le contrat n°5 (M2 -> M5).

    Exemple de sortie :
    `{"P1": {"priorité": 1, "usi": "oui", "spécialité": "Cardiology",
             "type_lit": "ICU", "salle_examen": "oui", ...}}`
    """
    all_facts = kb.facts
    out: dict[str, dict[str, Any]] = {}

    for pid in _patient_ids(kb, only_waiting=only_waiting):
        prio_1_4 = _effective_priority_1_to_4(all_facts, pid)
        needs_icu = ("RequiresICU", pid) in all_facts
        needs_ed_bed = ("RequiresEmergencyBed", pid) in all_facts
        needs_exam = ("RequiresExamRoom", pid) in all_facts
        needs_urgent = ("RequiresUrgentEvaluation", pid) in all_facts
        specs = _ordered_specialties(all_facts, pid)
        bed_type = "ICU" if needs_icu else ("Emergency" if needs_ed_bed else None)

        out[pid] = {
            "priorité": prio_1_4,
            "usi": "oui" if needs_icu else "non",
            "spécialité": specs[0] if specs else None,
            "spécialités": specs,
            "type_lit": bed_type,
            "salle_examen": "oui" if needs_exam else "non",
            "évaluation_urgente": "oui" if needs_urgent else "non",
        }
    return out


def export_search_requirements(kb: KnowledgeBase, only_waiting: bool = True) -> dict[str, dict[str, Any]]:
    """Adaptateur au format attendu par `agent.py` / `stubs.search_allocation`."""
    all_facts = kb.facts
    out: dict[str, dict[str, Any]] = {}
    for pid in _patient_ids(kb, only_waiting=only_waiting):
        icu = ("RequiresICU", pid) in all_facts
        specs = _ordered_specialties(all_facts, pid)
        prio = 3 if ("CriticalRisk", pid) in all_facts else 2 if ("HighRisk", pid) in all_facts else 1
        bed_type = "ICU" if icu else ("Emergency" if ("RequiresEmergencyBed", pid) in all_facts else None)
        out[pid] = {
            "priority": prio,
            "icu": icu,
            "specialty": specs[0] if specs else None,
            "specialties": specs,
            "bed_type": bed_type,
            "exam_room": ("RequiresExamRoom", pid) in all_facts,
        }
    return out


__all__ = ["export_requirements", "export_search_requirements", "reason"]


def run_bloc2_demo() -> None:
    """Démonstration exécutable complète du Bloc 2 sur le scénario P1/P2/P3 + événements."""
    import stubs as S
    from agent import Agent
    from facts import state_to_facts
    from interface import load_scenario
    from resolution import resolve_query

    print("=== DEMO BLOC 2 (simulation, sans usage médical réel) ===")
    state = load_scenario("scenarios/scenario_complet.json")
    facts = state_to_facts(state)
    for p in state.waiting_patients():
        ml_lvl, ml_p = S.ml_predict(p)
        facts.append(("ML_Risk", p.id, ml_lvl, ml_p))
        dl = S.dl_predict(p.history)
        if dl:
            facts.append(("DL_Risk", p.id, dl[0], dl[1]))

    kb = KnowledgeBase()
    kb.load_state_facts(facts)
    _, trace = kb.recompute()
    print("\n--- Trace chaînage avant + pont ML/DL ---")
    for line in trace:
        print(" ", line)

    print("\n--- Export exigences Membre 5 ---")
    print(export_requirements(kb))

    print("\n--- Résolution par réfutation : RequiresICU(P1)? ---")
    res = resolve_query(("RequiresICU", "P1"), facts)
    print(res.format_trace(only_proof_path=True))

    print("\n--- Résolution par réfutation : RequiresICU(P2)? ---")
    res_p2 = resolve_query(("RequiresICU", "P2"), facts)
    print(res_p2.format_trace(only_proof_path=True))

    print("\n--- Exécution orchestrateur Agent (Membre 1) avec moteur Membre 2 ---")
    agent = Agent(load_scenario("scenarios/scenario_complet.json"), reasoner=reason)
    agent.cycle()
    agent.execute_next()
    agent.execute_next()
    agent.handle_event({"type": "resource_unavailable", "id": "ICU1"})
    agent.run_plan()
    print("Etat final :", agent.state.summary())


if __name__ == "__main__":
    run_bloc2_demo()
