"""Moteur de déduction logique : chaînage avant (point fixe) et chaînage arrière (Bloc 2 - Membre 2).

Avertissement : ce module fait partie d'un prototype pédagogique de simulation
d'aide à la décision aux urgences, sans aucun usage médical réel.

Utilise l'algorithme d'unification (`unification.py`) sur les règles de Horn
(`rules.py`) et enregistre pour chaque fait dérivé sa justification dans la
`KnowledgeBase` (`kb.py`) afin de produire des traces lisibles en console et Markdown.
"""
from __future__ import annotations

from typing import Any, Iterable

from kb import Justification, KnowledgeBase
from rules import DEFAULT_RULES, Guard, Rule
from unification import (
    Substitution,
    apply_subst,
    format_subst,
    format_term,
    rename_vars,
    unify,
)


def _match_premises(
    premises: tuple[tuple[Any, ...], ...],
    known_facts: list[tuple[Any, ...]],
    guards: tuple[Guard, ...],
    subst: Substitution | None = None,
    idx: int = 0,
    matched: tuple[tuple[Any, ...], ...] = (),
) -> list[tuple[Substitution, tuple[tuple[Any, ...], ...]]]:
    """Trouve toutes les substitutions satisfaisant les prémisses et gardes d'une règle."""
    if subst is None:
        subst = {}
    if idx == len(premises):
        if all(g.holds(subst) for g in guards):
            return [(subst, matched)]
        return []

    target = premises[idx]
    solutions: list[tuple[Substitution, tuple[tuple[Any, ...], ...]]] = []
    # Filtrage rapide sur le nom du prédicat et l'arité pour éviter les unifications inutiles
    pred_name = target[0] if target else None
    arity = len(target)
    for fact in known_facts:
        if len(fact) != arity or fact[0] != pred_name:
            continue
        next_sub = unify(target, fact, subst)
        if next_sub is not None:
            solutions.extend(
                _match_premises(
                    premises,
                    known_facts,
                    guards,
                    next_sub,
                    idx + 1,
                    matched + (fact,),
                )
            )
    return solutions


def forward_chain(
    kb: KnowledgeBase,
    max_iterations: int = 50,
    include_iteration_headers: bool = False,
) -> tuple[set[tuple[Any, ...]], list[str]]:
    """Exécute le chaînage avant jusqu'au point fixe sur `kb`.

    Chaque conclusion dérivée conserve sa justification (règle, prémisses concrètes,
    substitution) pour l'explicabilité et le maintien de cohérence.

    Retourne `(faits_derives, trace_lisible)`.
    """
    trace: list[str] = []
    for it in range(1, max_iterations + 1):
        snapshot = kb.all_facts_sorted()
        new_in_iter: list[str] = []
        added_new_fact = False

        for rule in kb.rules:
            matches = _match_premises(rule.premises, snapshot, rule.guards)
            for subst, used_premises in matches:
                concl = apply_subst(rule.conclusion, subst)
                just = Justification(
                    rule_id=rule.rule_id,
                    premises=used_premises,
                    substitution_str=format_subst(subst),
                )
                is_new = kb.add_derived_fact(concl, just)
                if is_new:
                    added_new_fact = True
                    new_in_iter.append(just.format_line(concl))

        if new_in_iter:
            if include_iteration_headers:
                trace.append(f"--- Itération {it} ({len(new_in_iter)} nouveaux faits) ---")
            trace.extend(new_in_iter)

        if not added_new_fact:
            break

    kb.trace_log.extend(trace)
    return set(kb.derived_facts.keys()), trace


def backward_chain(
    goal: tuple[Any, ...],
    facts: Iterable[tuple[Any, ...]],
    rules: Iterable[Rule] | None = None,
    max_depth: int = 8,
) -> tuple[bool, list[tuple[Substitution, list[str]]]]:
    """Chaînage arrière (optionnel) pour prouver `goal` depuis `facts` et `rules`.

    Retourne `(True, [(substitution, trace_etapes), ...])` si au moins une preuve existe,
    sinon `(False, [])`.
    """
    fact_list = sorted({tuple(f) for f in facts}, key=lambda f: tuple(str(x) for x in f))
    rule_list = list(DEFAULT_RULES if rules is None else rules)
    counter = [0]

    def _prove(
        subgoal: tuple[Any, ...],
        subst: Substitution,
        depth: int,
        visited_goals: frozenset[tuple[Any, ...]],
    ) -> list[tuple[Substitution, list[str]]]:
        resolved_goal = apply_subst(subgoal, subst)
        if resolved_goal in visited_goals or depth > max_depth:
            return []

        results: list[tuple[Substitution, list[str]]] = []
        # 1) Unification directe avec un fait connu
        for f in fact_list:
            if len(f) == len(resolved_goal) and f[0] == resolved_goal[0]:
                s = unify(resolved_goal, f, subst)
                if s is not None:
                    results.append((s, [f"Fait établi : {format_term(f)}"]))

        # 2) Chaînage via la tête d'une règle
        next_visited = visited_goals | {resolved_goal}
        for rule in rule_list:
            if len(rule.conclusion) != len(resolved_goal) or rule.conclusion[0] != resolved_goal[0]:
                continue
            counter[0] += 1
            suffix = f"_bc{counter[0]}"
            (renamed_premises, renamed_head), mapping = rename_vars(
                (rule.premises, rule.conclusion), suffix
            )
            s_head = unify(resolved_goal, renamed_head, subst)
            if s_head is None:
                continue

            # Prouver récursivement chaque prémisse de gauche à droite
            partial_proofs: list[tuple[Substitution, list[str]]] = [(s_head, [])]
            for prem in renamed_premises:
                next_partials: list[tuple[Substitution, list[str]]] = []
                for cur_sub, cur_trace in partial_proofs:
                    for sub_out, sub_trace in _prove(prem, cur_sub, depth + 1, next_visited):
                        next_partials.append((sub_out, cur_trace + sub_trace))
                partial_proofs = next_partials
                if not partial_proofs:
                    break

            for final_sub, prem_trace in partial_proofs:
                # Vérifier les éventuelles gardes sur la règle renommée
                mapped_sub = {v: apply_subst(apply_subst(v, mapping), final_sub) for v in mapping}
                if all(g.holds(mapped_sub) for g in rule.guards):
                    concl_inst = apply_subst(resolved_goal, final_sub)
                    step_line = f"[{rule.rule_id}] => {format_term(concl_inst)}"
                    results.append((final_sub, prem_trace + [step_line]))

        return results

    proofs = _prove(goal, {}, 0, frozenset())
    return (len(proofs) > 0), proofs


def format_inference_markdown(kb: KnowledgeBase, title: str = "Trace d'inférence (chaînage avant)") -> str:
    """Exporte les faits dérivés et leurs justifications au format Markdown."""
    lines = [
        f"# {title}",
        "",
        "> Simulation pédagogique d'aide à la décision aux urgences, sans usage médical réel.",
        "",
        f"- Faits de base : **{len(kb.base_facts)}**",
        f"- Faits dérivés : **{len(kb.derived_facts)}**",
        "",
        "## Étapes d'inférence",
        "",
    ]
    if not kb.trace_log:
        lines.append("_Aucune nouvelle dérivation._")
    else:
        for i, entry in enumerate(kb.trace_log, start=1):
            lines.append(f"{i}. `{entry}`")
    return "\n".join(lines) + "\n"
