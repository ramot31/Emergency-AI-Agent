"""Base de connaissances (KB) du moteur logique (Bloc 2 - Membre 2).

Avertissement : ce module fait partie d'un prototype pédagogique de simulation
d'aide à la décision aux urgences, sans aucun usage médical réel.

Stocke :
- les faits de base (issus de l'environnement, des prédictions ML/DL ou ajoutés),
- les règles logiques (clauses de Horn),
- les faits dérivés et leurs justifications pour l'explicabilité et le maintien
  de cohérence lors des ajouts/retraits (`assert_fact`, `retract_fact`, `update`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from environment import HR_HIGH, SBP_HIGH, SBP_LOW, SPO2_LOW, TEMP_FEVER
from rules import DEFAULT_RULES, Rule
from unification import format_term

# Normalisation tolérante des variantes de casse de symptômes (ex. event_saisie.json)
SYMPTOM_ALIASES: dict[str, str] = {
    "chestpain": "ChestPain",
    "shortnessofbreath": "ShortnessOfBreath",
    "abdominalpain": "AbdominalPain",
    "fever": "Fever",
    "confusion": "Confusion",
}


def normalize_fact(fact: tuple[Any, ...]) -> tuple[Any, ...]:
    """Normalise les variantes connues de casse sur `HasSymptom` sans altérer le reste."""
    if len(fact) == 3 and fact[0] == "HasSymptom" and isinstance(fact[2], str):
        canon = SYMPTOM_ALIASES.get(fact[2].lower())
        if canon:
            return ("HasSymptom", fact[1], canon)
    return tuple(fact)


@dataclass(frozen=True)
class Justification:
    """Trace d'obtention d'un fait dérivé (règle/source, prémisses utilisées, substitution)."""

    rule_id: str
    premises: tuple[tuple[Any, ...], ...]
    substitution_str: str
    note: str = ""

    def format_line(self, conclusion: tuple[Any, ...]) -> str:
        prem_str = " ^ ".join(format_term(p) for p in self.premises) if self.premises else "bridge"
        suffix = f" ({self.note})" if self.note else ""
        return f"[{self.rule_id}] {prem_str} -> {format_term(conclusion)} {self.substitution_str}{suffix}".strip()


class KnowledgeBase:
    """Base de connaissances symbolique avec gestion des justifications."""

    def __init__(self, rules: Iterable[Rule] | None = None):
        self.rules: list[Rule] = list(DEFAULT_RULES if rules is None else rules)
        self.base_facts: dict[tuple[Any, ...], str] = {}
        self.derived_facts: dict[tuple[Any, ...], list[Justification]] = {}
        self.trace_log: list[str] = []

    @property
    def facts(self) -> set[tuple[Any, ...]]:
        """Ensemble de tous les faits actuellement vrais (base + dérivés)."""
        return set(self.base_facts.keys()) | set(self.derived_facts.keys())

    def all_facts_sorted(self) -> list[tuple[Any, ...]]:
        """Retourne tous les faits vrais triés de façon déterministe."""
        return sorted(self.facts, key=lambda f: tuple(str(x) for x in f))

    def contains(self, fact: tuple[Any, ...]) -> bool:
        norm = normalize_fact(fact)
        return norm in self.base_facts or norm in self.derived_facts

    def assert_fact(self, fact: tuple[Any, ...], source: str = "environment") -> bool:
        """Ajoute un fait de base dans la KB. Retourne True s'il est nouveau."""
        norm = normalize_fact(fact)
        if norm in self.base_facts:
            return False
        self.base_facts[norm] = source
        return True

    def assert_facts(self, facts: Iterable[tuple[Any, ...]], source: str = "environment") -> int:
        """Ajoute plusieurs faits de base et retourne le nombre de nouveaux faits."""
        added = 0
        for f in facts:
            if self.assert_fact(f, source=source):
                added += 1
        return added

    def add_derived_fact(self, fact: tuple[Any, ...], justification: Justification) -> bool:
        """Enregistre un fait dérivé et sa justification. Retourne True si le fait est nouveau."""
        norm = normalize_fact(fact)
        is_new = norm not in self.derived_facts and norm not in self.base_facts
        justs = self.derived_facts.setdefault(norm, [])
        if justification not in justs:
            justs.append(justification)
        return is_new

    def retract_fact(self, fact: tuple[Any, ...]) -> set[tuple[Any, ...]]:
        """Retire un fait de base et invalide en cascade les dérivés sans support valide."""
        norm = normalize_fact(fact)
        removed: set[tuple[Any, ...]] = set()
        if norm in self.base_facts:
            del self.base_facts[norm]
            removed.add(norm)
        removed |= self._propagate_retraction({norm})
        return removed

    def _propagate_retraction(self, seeds: set[tuple[Any, ...]]) -> set[tuple[Any, ...]]:
        """Maintien de cohérence (TMS) : supprime les faits dérivés sans justification valide."""
        invalidated: set[tuple[Any, ...]] = set()
        frontier = set(seeds)
        while frontier:
            lost = frontier.pop()
            for dfact in list(self.derived_facts.keys()):
                remaining = [
                    j for j in self.derived_facts[dfact]
                    if lost not in j.premises and all(p in self.base_facts or p in self.derived_facts for p in j.premises)
                ]
                if remaining:
                    self.derived_facts[dfact] = remaining
                else:
                    del self.derived_facts[dfact]
                    if dfact not in self.base_facts:
                        invalidated.add(dfact)
                        frontier.add(dfact)
        return invalidated

    def explain(self, fact: tuple[Any, ...]) -> list[str]:
        """Retourne la trace explicative d'un fait (de base ou dérivé)."""
        norm = normalize_fact(fact)
        if norm in self.base_facts and norm not in self.derived_facts:
            return [f"[BASE:{self.base_facts[norm]}] {format_term(norm)}"]
        if norm not in self.derived_facts:
            return [f"[INCONNU] {format_term(norm)}"]
        lines: list[str] = []
        visited: set[tuple[Any, ...]] = set()
        self._collect_explanation(norm, lines, visited)
        return lines

    def _collect_explanation(
        self,
        fact: tuple[Any, ...],
        lines: list[str],
        visited: set[tuple[Any, ...]],
    ) -> None:
        if fact in visited:
            return
        visited.add(fact)
        if fact in self.base_facts and fact not in self.derived_facts:
            lines.append(f"  fait de base ({self.base_facts[fact]}) : {format_term(fact)}")
            return
        justs = self.derived_facts.get(fact, [])
        if not justs:
            return
        j = justs[0]
        for prem in j.premises:
            self._collect_explanation(prem, lines, visited)
        lines.append(f"  {j.format_line(fact)}")

    def clear_derived(self) -> None:
        """Efface uniquement les faits dérivés (utile avant recalcul complet)."""
        self.derived_facts.clear()
        self.trace_log.clear()

    def load_state_facts(self, facts: Iterable[tuple[Any, ...]]) -> None:
        """Charge un instantané complet de faits (environnement + ML/DL)."""
        self.base_facts.clear()
        self.derived_facts.clear()
        self.trace_log.clear()
        for f in facts:
            src = "ml" if f and f[0] == "ML_Risk" else "dl" if f and f[0] == "DL_Risk" else "environment"
            self.assert_fact(f, source=src)

    def recompute(self) -> tuple[set[tuple[Any, ...]], list[str]]:
        """Recalcule le pont ML/DL puis le chaînage avant jusqu'au point fixe."""
        from inference import forward_chain
        from ml_dl_bridge import apply_ml_dl_bridge
        from risk_judgment import trace_rule_ml_judgment

        # La modération ML dépend des faits actuels : recalculer les dérivés
        # évite de conserver une hausse devenue obsolète après mise à jour.
        self.clear_derived()
        d_bridge, tr_bridge = apply_ml_dl_bridge(self)
        d_chain, tr_chain = forward_chain(self)
        judgment_trace = trace_rule_ml_judgment(self)
        self.trace_log.extend(judgment_trace)
        return (d_bridge | d_chain), (tr_bridge + tr_chain + judgment_trace)

    def _retract_matching(self, predicate: str, patient_id: str) -> set[tuple[Any, ...]]:
        """Retire tous les faits de base d'un patient correspondant à `predicate`."""
        targets = [
            f for f in list(self.base_facts.keys())
            if len(f) >= 2 and f[0] == predicate and f[1] == patient_id
        ]
        removed: set[tuple[Any, ...]] = set()
        for t in targets:
            removed |= self.retract_fact(t)
        return removed

    def _assert_vital_discretizations(
        self,
        pid: str,
        hr: int,
        bp: int,
        spo2: int,
        temperature: float | None = None,
    ) -> list[tuple[Any, ...]]:
        """Émet les prédicats discrétisés en réutilisant strictement les seuils du Membre 1."""
        added: list[tuple[Any, ...]] = []
        candidates: list[tuple[Any, ...]] = []
        if bp < SBP_LOW:
            candidates.append(("LowBloodPressure", pid))
        elif bp > SBP_HIGH:
            candidates.append(("HighBloodPressure", pid))
        if spo2 < SPO2_LOW:
            candidates.append(("LowOxygen", pid))
        if hr > HR_HIGH:
            candidates.append(("Tachycardia", pid))
        if temperature is not None and temperature >= TEMP_FEVER:
            candidates.append(("Fever", pid))
        for fact in candidates:
            if self.assert_fact(fact, source="environment"):
                added.append(fact)
        return added

    def update(self, event: dict[str, Any], recompute: bool = True) -> dict[str, Any]:
        """Met à jour la KB sur événement dynamique avec maintien de cohérence (TMS).

        Compatible avec les événements du Membre 1 (`resource_unavailable`,
        `resource_available`, `new_patient`, `new_measurement`), l'alias du contrat PDF
        (`icu_unavailable`), et les mises à jour de prédictions (`ml_prediction`,
        `dl_prediction`).
        """
        ev_type = event["type"]
        retracted: set[tuple[Any, ...]] = set()
        added: list[tuple[Any, ...]] = []

        if ev_type in ("resource_unavailable", "icu_unavailable"):
            rid = event["id"]
            retracted |= self.retract_fact(("Available", rid))

        elif ev_type == "resource_available":
            rid = event["id"]
            f = ("Available", rid)
            if self.assert_fact(f, source="environment"):
                added.append(f)

        elif ev_type == "new_patient":
            p = event["patient"]
            pid = p["id"]
            base = [
                ("Patient", pid),
                ("Age", pid, int(p["age"])),
                ("HeartRate", pid, int(p["heart_rate"])),
                ("OxygenSaturation", pid, int(p["spo2"])),
                ("Temperature", pid, float(p.get("temperature", 37.0))),
                ("RespiratoryRate", pid, int(p.get("resp_rate", 16))),
                ("PatientWaiting" if p.get("status", "waiting") == "waiting" else "UnderTreatment", pid),
            ]
            for sym in p.get("symptoms", []):
                base.append(("HasSymptom", pid, sym))
            for f in base:
                if self.assert_fact(f, source="environment"):
                    added.append(f)
            added.extend(
                self._assert_vital_discretizations(
                    pid,
                    int(p["heart_rate"]),
                    int(p["systolic_bp"]),
                    int(p["spo2"]),
                    float(p.get("temperature", 37.0)),
                )
            )

        elif ev_type == "new_measurement":
            pid = event["patient"]
            hr, bp, spo2 = int(event["hr"]), int(event["bp"]), int(event["spo2"])
            for pred in (
                "HeartRate",
                "OxygenSaturation",
                "LowBloodPressure",
                "HighBloodPressure",
                "LowOxygen",
                "Tachycardia",
            ):
                retracted |= self._retract_matching(pred, pid)
            for f in (("HeartRate", pid, hr), ("OxygenSaturation", pid, spo2)):
                if self.assert_fact(f, source="environment"):
                    added.append(f)
            added.extend(self._assert_vital_discretizations(pid, hr, bp, spo2))

        elif ev_type in ("ml_prediction", "dl_prediction"):
            pid = event["patient"]
            pred_name = "ML_Risk" if ev_type == "ml_prediction" else "DL_Risk"
            src = "ml" if ev_type == "ml_prediction" else "dl"
            retracted |= self._retract_matching(pred_name, pid)
            f = (pred_name, pid, str(event["level"]), float(event["prob"]))
            if self.assert_fact(f, source=src):
                added.append(f)

        else:
            raise ValueError(f"Événement inconnu pour la KB : {ev_type}")

        new_derived: set[tuple[Any, ...]] = set()
        trace: list[str] = []
        if recompute:
            new_derived, trace = self.recompute()

        return {
            "event_type": ev_type,
            "retracted_or_invalidated": sorted(retracted, key=lambda x: tuple(str(i) for i in x)),
            "added": added,
            "derived_count": len(new_derived),
            "trace": trace,
        }


def reason(facts: Iterable[tuple[Any, ...]]) -> tuple[set[tuple[Any, ...]], list[str]]:
    """Point d'entrée compatible avec le contrat `stubs.reason(facts)` de `agent.py`.

    Entrée : liste de faits (tuples produits par `state_to_facts` + `ML_Risk`/`DL_Risk`).
    Sortie : `(faits_derives, trace)` où `faits_derives` est un ensemble de tuples et
    `trace` une liste de lignes explicatives.
    """
    kb = KnowledgeBase()
    kb.load_state_facts(facts)
    derived, trace = kb.recompute()
    return derived, trace
