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
