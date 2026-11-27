"""Moteur de résolution du premier ordre par réfutation et mise en FNC (Bloc 2 - Membre 2).

Avertissement : ce module fait partie d'un prototype pédagogique de simulation
d'aide à la décision aux urgences, sans aucun usage médical réel.

Implémente sans bibliothèque externe :
1. Représentation de formules du premier ordre (`Atom`, `Not`, `And`, `Or`,
   `Implies`, `Iff`, `Forall`, `Exists`).
2. Conversion complète en Forme Normale Conjonctive (FNC) :
   - élimination des équivalences et implications,
   - descente des négations (Forme Normale Négative),
   - standardisation des variables,
   - skolémisation des quantificateurs existentiels,
   - suppression des quantificateurs universels et distribution de `∨` sur `∧`.
3. Résolution binaire par réfutation avec unification, stratégie *Set of Support*
   + préférence unitaire et limite d'étapes/profondeur contre l'explosion combinatoire.
4. Production de traces pas à pas (clauses parentes, unificateur, résolvant, `□`)
   en console et en Markdown.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Iterable

from rules import DEFAULT_RULES, Rule
from unification import (
    Compound,
    Substitution,
    Var,
    apply_subst,
    collect_vars,
    format_subst,
    format_term,
    rename_vars,
    unify,
)


# ---------------------------------------------------------------------------
# 1. Syntaxe abstraite des formules du premier ordre
# ---------------------------------------------------------------------------

class Formula:
    """Classe de base pour les formules logiques du premier ordre."""


@dataclass(frozen=True)
class Atom(Formula):
    atom: tuple[Any, ...]

    def __repr__(self) -> str:
        return format_term(self.atom)


@dataclass(frozen=True)
class Not(Formula):
    inner: Formula

    def __repr__(self) -> str:
        return f"¬{self.inner}"


@dataclass(frozen=True)
class And(Formula):
    left: Formula
    right: Formula

    def __repr__(self) -> str:
        return f"({self.left} ∧ {self.right})"


@dataclass(frozen=True)
class Or(Formula):
    left: Formula
    right: Formula

    def __repr__(self) -> str:
        return f"({self.left} ∨ {self.right})"


@dataclass(frozen=True)
class Implies(Formula):
    premise: Formula
    conclusion: Formula

    def __repr__(self) -> str:
        return f"({self.premise} → {self.conclusion})"


@dataclass(frozen=True)
class Iff(Formula):
    left: Formula
    right: Formula

    def __repr__(self) -> str:
        return f"({self.left} ↔ {self.right})"


@dataclass(frozen=True)
class Forall(Formula):
    var: Var
    body: Formula

    def __repr__(self) -> str:
        return f"∀{self.var.name} ({self.body})"


@dataclass(frozen=True)
class Exists(Formula):
    var: Var
    body: Formula

    def __repr__(self) -> str:
        return f"∃{self.var.name} ({self.body})"


# ---------------------------------------------------------------------------
# 2. Littéraux, Clauses et Transformation en FNC
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Literal:
    """Littéral atomique positif (`negated=False`) ou négatif (`negated=True`)."""

    atom: tuple[Any, ...]
    negated: bool = False

    def negate(self) -> "Literal":
        return Literal(self.atom, not self.negated)

    def apply(self, subst: Substitution) -> "Literal":
        return Literal(apply_subst(self.atom, subst), self.negated)

    def __repr__(self) -> str:
        prefix = "¬" if self.negated else ""
        return f"{prefix}{format_term(self.atom)}"


@dataclass(frozen=True)
class Clause:
    """Disjonction de littéraux ; clause vide = `literals == ()` (`□`)."""

    literals: tuple[Literal, ...]
    source: str = ""

    @property
    def is_empty(self) -> bool:
        return len(self.literals) == 0

    def is_tautology(self) -> bool:
        pos = {lit.atom for lit in self.literals if not lit.negated}
        neg = {lit.atom for lit in self.literals if lit.negated}
        return bool(pos & neg)

    def __repr__(self) -> str:
        if self.is_empty:
            return "□"
        return " ∨ ".join(repr(lit) for lit in self.literals)


def _eliminate_implications(f: Formula) -> Formula:
    if isinstance(f, Atom):
        return f
    if isinstance(f, Not):
        return Not(_eliminate_implications(f.inner))
    if isinstance(f, And):
        return And(_eliminate_implications(f.left), _eliminate_implications(f.right))
    if isinstance(f, Or):
        return Or(_eliminate_implications(f.left), _eliminate_implications(f.right))
    if isinstance(f, Implies):
        return Or(Not(_eliminate_implications(f.premise)), _eliminate_implications(f.conclusion))
    if isinstance(f, Iff):
        a = _eliminate_implications(f.left)
        b = _eliminate_implications(f.right)
        return And(Or(Not(a), b), Or(Not(b), a))
    if isinstance(f, Forall):
        return Forall(f.var, _eliminate_implications(f.body))
    if isinstance(f, Exists):
        return Exists(f.var, _eliminate_implications(f.body))
    raise TypeError(f"Formule inconnue : {f!r}")


def _push_negations(f: Formula) -> Formula:
    """Met en Forme Normale Négative (FNN)."""
    if isinstance(f, Atom):
        return f
    if isinstance(f, And):
        return And(_push_negations(f.left), _push_negations(f.right))
    if isinstance(f, Or):
        return Or(_push_negations(f.left), _push_negations(f.right))
    if isinstance(f, Forall):
        return Forall(f.var, _push_negations(f.body))
    if isinstance(f, Exists):
        return Exists(f.var, _push_negations(f.body))
    if isinstance(f, Not):
        g = f.inner
        if isinstance(g, Atom):
            return f
        if isinstance(g, Not):
            return _push_negations(g.inner)
        if isinstance(g, And):
            return Or(_push_negations(Not(g.left)), _push_negations(Not(g.right)))
        if isinstance(g, Or):
            return And(_push_negations(Not(g.left)), _push_negations(Not(g.right)))
        if isinstance(g, Forall):
            return Exists(g.var, _push_negations(Not(g.body)))
        if isinstance(g, Exists):
            return Forall(g.var, _push_negations(Not(g.body)))
    raise TypeError(f"Impossible de descendre la négation sur : {f!r}")


def _subst_formula(f: Formula, subst: Substitution) -> Formula:
    if isinstance(f, Atom):
        return Atom(apply_subst(f.atom, subst))
    if isinstance(f, Not):
        return Not(_subst_formula(f.inner, subst))
    if isinstance(f, And):
        return And(_subst_formula(f.left, subst), _subst_formula(f.right, subst))
    if isinstance(f, Or):
        return Or(_subst_formula(f.left, subst), _subst_formula(f.right, subst))
    if isinstance(f, Forall):
        sub2 = {k: v for k, v in subst.items() if k != f.var}
        return Forall(f.var, _subst_formula(f.body, sub2))
    if isinstance(f, Exists):
        sub2 = {k: v for k, v in subst.items() if k != f.var}
        return Exists(f.var, _subst_formula(f.body, sub2))
    raise TypeError(f)


def _standardize_and_skolemize(
    f: Formula,
    univ_vars: tuple[Var, ...],
    var_counter: list[int],
    sk_counter: list[int],
) -> Formula:
    """Standardise les variables liées, skolémise les `∃` et supprime les `∀`."""
    if isinstance(f, Atom):
        return f
    if isinstance(f, Not):
        return Not(_standardize_and_skolemize(f.inner, univ_vars, var_counter, sk_counter))
    if isinstance(f, And):
        return And(
            _standardize_and_skolemize(f.left, univ_vars, var_counter, sk_counter),
            _standardize_and_skolemize(f.right, univ_vars, var_counter, sk_counter),
        )
    if isinstance(f, Or):
        return Or(
            _standardize_and_skolemize(f.left, univ_vars, var_counter, sk_counter),
            _standardize_and_skolemize(f.right, univ_vars, var_counter, sk_counter),
        )
    if isinstance(f, Forall):
        var_counter[0] += 1
        fresh = Var(f"{f.var.name}_{var_counter[0]}")
        body = _subst_formula(f.body, {f.var: fresh})
        return _standardize_and_skolemize(body, univ_vars + (fresh,), var_counter, sk_counter)
    if isinstance(f, Exists):
        sk_counter[0] += 1
        sk_name = f"sk_{sk_counter[0]}"
        sk_term: Any = Compound(sk_name, univ_vars) if univ_vars else sk_name
        body = _subst_formula(f.body, {f.var: sk_term})
        return _standardize_and_skolemize(body, univ_vars, var_counter, sk_counter)
    raise TypeError(f)


def _distribute_or(a: Formula, b: Formula) -> Formula:
    if isinstance(a, And):
        return And(_distribute_or(a.left, b), _distribute_or(a.right, b))
    if isinstance(b, And):
        return And(_distribute_or(a, b.left), _distribute_or(a, b.right))
    return Or(a, b)


def _to_cnf_formula(f: Formula) -> Formula:
    if isinstance(f, And):
        return And(_to_cnf_formula(f.left), _to_cnf_formula(f.right))
    if isinstance(f, Or):
        return _distribute_or(_to_cnf_formula(f.left), _to_cnf_formula(f.right))
    return f


def _extract_clauses(f: Formula, source: str) -> list[Clause]:
    if isinstance(f, And):
        return _extract_clauses(f.left, source) + _extract_clauses(f.right, source)

    def _collect_disjuncts(node: Formula) -> list[Literal]:
        if isinstance(node, Or):
            return _collect_disjuncts(node.left) + _collect_disjuncts(node.right)
        if isinstance(node, Not) and isinstance(node.inner, Atom):
            return [Literal(node.inner.atom, negated=True)]
        if isinstance(node, Atom):
            return [Literal(node.atom, negated=False)]
        raise ValueError(f"Noeud inattendu dans clause FNC : {node!r}")

    lits = tuple(dict.fromkeys(_collect_disjuncts(f)))
    return [Clause(lits, source=source)]


def to_cnf(formula: Formula, source: str = "formula") -> list[Clause]:
    """Convertit une formule du premier ordre quelconque en liste de clauses FNC."""
    no_imp = _eliminate_implications(formula)
    nnf = _push_negations(no_imp)
    skolemized = _standardize_and_skolemize(nnf, (), [0], [0])
    cnf_f = _to_cnf_formula(skolemized)
    return _extract_clauses(cnf_f, source=source)


def rule_to_formula(rule: Rule) -> Formula:
    """Convertit une règle de Horn sans garde en formule FOL `∀vars (P1 ∧ ... ∧ Pn → C)`."""
    if not rule.premises:
        body: Formula = Atom(rule.conclusion)
    else:
        conj: Formula = Atom(rule.premises[0])
        for p in rule.premises[1:]:
            conj = And(conj, Atom(p))
        body = Implies(conj, Atom(rule.conclusion))
    vars_in_rule = sorted(collect_vars((rule.premises, rule.conclusion)), key=lambda v: v.name)
    for v in reversed(vars_in_rule):
        body = Forall(v, body)
    return body


def ground_guarded_rules(facts: Iterable[tuple[Any, ...]], rules: Iterable[Rule]) -> list[Clause]:
    """Génère les clauses instanciées des règles comportant des gardes numériques (ex. R01-R03)."""
    fact_list = list(facts)
    clauses: list[Clause] = []
    for rule in rules:
        if rule.is_pure_horn:
            continue
        # Règles à 1 prémisse avec garde arithmétique (Age, RespiratoryRate, OxygenSaturation)
        if len(rule.premises) == 1:
            prem = rule.premises[0]
            for f in fact_list:
                if len(f) == len(prem) and f[0] == prem[0]:
                    sub = unify(prem, f)
                    if sub is not None and all(g.holds(sub) for g in rule.guards):
                        inst_prem = apply_subst(prem, sub)
                        inst_concl = apply_subst(rule.conclusion, sub)
                        clauses.append(
                            Clause(
                                (Literal(inst_prem, negated=True), Literal(inst_concl, negated=False)),
                                source=f"{rule.rule_id}(inst)",
                            )
                        )
    return clauses


# ---------------------------------------------------------------------------
# 3. Résolution par réfutation avec unification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolutionStep:
    """Une étape élémentaire de résolution binaire entre deux clauses parentes."""

    step_num: int
    resolvent_id: str
    parent1_id: str
    parent2_id: str
    literal1: Literal
    literal2: Literal
    unifier: Substitution
    resolvent: Clause
    depth: int

    def format_line(self) -> str:
        return (
            f"Étape {self.step_num} [{self.resolvent_id}] : "
            f"{self.parent1_id} + {self.parent2_id} "
            f"sur ({self.literal1} , {self.literal2}) "
            f"unif={format_subst(self.unifier)} => {self.resolvent}"
        )


@dataclass
class ResolutionResult:
    """Résultat complet d'une preuve par réfutation."""

    proved: bool
    query: tuple[Any, ...]
    initial_clauses: list[tuple[str, Clause, bool]]  # (id, clause, is_sos)
    steps: list[ResolutionStep]
    proof_steps: list[ResolutionStep]
    explored_steps: int
    stop_reason: str

    def format_trace(self, only_proof_path: bool = False) -> str:
        lines = [
            f"Requête : {format_term(self.query)}?",
            f"Résultat : {'PROUVÉ (clause vide □ atteinte)' if self.proved else 'NON PROUVÉ'} ({self.stop_reason})",
            f"Clauses initiales ({len(self.initial_clauses)}) :",
        ]
        for cid, cl, is_sos in self.initial_clauses:
            tag = " [SoS / ¬requête]" if is_sos else ""
            lines.append(f"  {cid}: {cl}{tag}")

        selected = self.proof_steps if (only_proof_path and self.proved) else self.steps
        label = "Chemin minimal de réfutation" if (only_proof_path and self.proved) else "Étapes de résolution"
        lines.append(f"{label} ({len(selected)}) :")
        for st in selected:
            lines.append("  " + st.format_line())
        return "\n".join(lines)

    def to_markdown(self, title: str | None = None) -> str:
        hdr = title or f"Trace de résolution : `{format_term(self.query)}?`"
        lines = [
            f"# {hdr}",
            "",
            "> Simulation pédagogique d'aide à la décision aux urgences, sans usage médical réel.",
            "",
            f"- **Requête** : `{format_term(self.query)}?`",
            f"- **Verdict** : **{'PROUVÉ (□)' if self.proved else 'NON PROUVÉ'}** — {self.stop_reason}",
            f"- **Étapes explorées** : {self.explored_steps}",
            "",
            "## Clauses initiales (FNC)",
            "",
        ]
        for cid, cl, is_sos in self.initial_clauses:
            badge = " *(négation de la requête — SoS)*" if is_sos else f" *({cl.source})*" if cl.source else ""
            lines.append(f"- `{cid}` : `{cl}`{badge}")
        lines.append("")
        if self.proved:
            lines.append("## Chemin de réfutation jusqu'à `□`")
            lines.append("")
            for st in self.proof_steps:
                lines.append(f"1. `{st.format_line()}`")
        else:
            lines.append("## Trace d'exploration (aucune réfutation)")
            lines.append("")
            if not self.steps:
                lines.append("_Aucun résolvant applicable depuis la négation de la requête._")
            else:
                for st in self.steps[:20]:
                    lines.append(f"1. `{st.format_line()}`")
        return "\n".join(lines) + "\n"


def _canonical_signature(clause: Clause) -> tuple[tuple[bool, str], ...]:
    """Signature canonique d'une clause modulo renommage des variables (anti-doublon)."""
    vars_ordered: list[Var] = []
    for lit in clause.literals:
        for v in sorted(collect_vars(lit.atom), key=lambda vv: vv.name):
            if v not in vars_ordered:
                vars_ordered.append(v)
    canon_map = {v: Var(f"_v{i}") for i, v in enumerate(vars_ordered)}
    norm_lits = sorted(
        ((lit.negated, format_term(apply_subst(lit.atom, canon_map))) for lit in clause.literals)
    )
    return tuple(norm_lits)


def _factor_clause(clause: Clause) -> Clause:
    """Supprime les littéraux dupliqués dans une clause."""
    unique = tuple(dict.fromkeys(clause.literals))
    return Clause(unique, source=clause.source)


def _resolve_pair(
    c1: Clause,
    c2: Clause,
    counter: list[int],
) -> list[tuple[Literal, Literal, Substitution, Clause]]:
    """Calcule tous les résolvants binaires possibles entre `c1` et `c2` après standardisation."""
    counter[0] += 1
    r1_lits = tuple(
        Literal(rename_vars(lit.atom, f"_a{counter[0]}")[0], lit.negated) for lit in c1.literals
    )
    r2_lits = tuple(
        Literal(rename_vars(lit.atom, f"_b{counter[0]}")[0], lit.negated) for lit in c2.literals
    )

    resolvents: list[tuple[Literal, Literal, Substitution, Clause]] = []
    for i, l1 in enumerate(r1_lits):
        for j, l2 in enumerate(r2_lits):
            if l1.negated == l2.negated:
                continue
            if len(l1.atom) != len(l2.atom) or l1.atom[0] != l2.atom[0]:
                continue
            subst = unify(l1.atom, l2.atom)
            if subst is None:
                continue
            rem1 = [r1_lits[k].apply(subst) for k in range(len(r1_lits)) if k != i]
            rem2 = [r2_lits[k].apply(subst) for k in range(len(r2_lits)) if k != j]
            new_clause = _factor_clause(Clause(tuple(rem1 + rem2)))
            if new_clause.is_tautology():
                continue
            resolvents.append((l1, l2, subst, new_clause))
    return resolvents


def _extract_proof_path(steps: list[ResolutionStep], empty_step: ResolutionStep) -> list[ResolutionStep]:
    """Extrait uniquement les étapes ancêtres menant à la clause vide `□`."""
    by_id = {st.resolvent_id: st for st in steps}
    needed: list[ResolutionStep] = []
    seen_ids: set[str] = set()

    def _visit(cid: str) -> None:
        if cid in seen_ids or cid not in by_id:
            return
        seen_ids.add(cid)
        st = by_id[cid]
        _visit(st.parent1_id)
        _visit(st.parent2_id)
        needed.append(st)

    _visit(empty_step.resolvent_id)
    return needed


def resolve_query(
    query: tuple[Any, ...],
    facts: Iterable[tuple[Any, ...]],
    rules: Iterable[Rule] | None = None,
    *,
    extra_clauses: Iterable[Clause] = (),
    max_steps: int = 300,
    max_depth: int = 12,
) -> ResolutionResult:
    """Prouve `query` par résolution par réfutation (FNC + négation de `query`).

    Utilise :
    - la stratégie *Set of Support* (tout résolvant dérive au moins d'une clause SoS),
    - une file de priorité favorisant les clauses les plus courtes (*unit preference*),
    - une limite sur `max_steps` et `max_depth` pour éviter l'explosion combinatoire.
    """
    fact_list = sorted({tuple(f) for f in facts}, key=lambda f: tuple(str(x) for x in f))
    rule_list = list(DEFAULT_RULES if rules is None else rules)

    all_clauses: list[tuple[str, Clause, bool]] = []
    seen_sigs: set[tuple[tuple[bool, str], ...]] = set()

    def _add_clause(cl: Clause, is_sos: bool) -> str | None:
        if cl.is_tautology():
            return None
        sig = _canonical_signature(cl)
        if sig in seen_sigs:
            return None
        seen_sigs.add(sig)
        cid = f"C{len(all_clauses) + 1}"
        all_clauses.append((cid, cl, is_sos))
        return cid

    # 1) Faits unitaires de la KB
    for f in fact_list:
        _add_clause(Clause((Literal(f, negated=False),), source="fait"), is_sos=False)

    # 2) Règles converties en FNC (et gardes numériques instanciées)
    for r in rule_list:
        if r.is_pure_horn:
            for cl in to_cnf(rule_to_formula(r), source=r.rule_id):
                _add_clause(cl, is_sos=False)
    for cl in ground_guarded_rules(fact_list, rule_list):
        _add_clause(cl, is_sos=False)

    # 3) Éventuelles clauses FNC supplémentaires
    for cl in extra_clauses:
        _add_clause(cl, is_sos=False)

    # 4) Négation de la requête (placée dans le Set of Support)
    neg_query_clause = Clause((Literal(query, negated=True),), source="¬requête")
    _add_clause(neg_query_clause, is_sos=True)

    initial_snapshot = list(all_clauses)
    clause_depth: dict[str, int] = {cid: 0 for cid, _, _ in all_clauses}
    # File de priorité : (longueur_clause, profondeur, indice_clause, cid, clause, is_sos)
    pq: list[tuple[int, int, int, str, Clause, bool]] = []
    for idx, (cid, cl, is_sos) in enumerate(all_clauses):
        if is_sos:
            heapq.heappush(pq, (len(cl.literals), 0, idx, cid, cl, True))

    steps: list[ResolutionStep] = []
    rename_counter = [0]
    step_num = 0

    while pq and step_num < max_steps:
        _, depth_a, _, cid_a, cl_a, _ = heapq.heappop(pq)
        if depth_a >= max_depth:
            continue

        # Résolution de la clause SoS courante contre toutes les clauses disponibles
        snapshot_len = len(all_clauses)
        for idx_b in range(snapshot_len):
            cid_b, cl_b, _ = all_clauses[idx_b]
            if cid_a == cid_b:
                continue
            new_depth = max(depth_a, clause_depth[cid_b]) + 1
            if new_depth > max_depth:
                continue

            for l1, l2, subst, resolvent in _resolve_pair(cl_a, cl_b, rename_counter):
                if resolvent.is_empty:
                    step_num += 1
                    rid = f"C{len(all_clauses) + 1}"
                    st = ResolutionStep(
                        step_num=step_num,
                        resolvent_id=rid,
                        parent1_id=cid_a,
                        parent2_id=cid_b,
                        literal1=l1,
                        literal2=l2,
                        unifier=subst,
                        resolvent=resolvent,
                        depth=new_depth,
                    )
                    steps.append(st)
                    proof = _extract_proof_path(steps, st)
                    return ResolutionResult(
                        proved=True,
                        query=query,
                        initial_clauses=initial_snapshot,
                        steps=steps,
                        proof_steps=proof,
                        explored_steps=step_num,
                        stop_reason=f"Clause vide atteinte en {step_num} étape(s)",
                    )

                sig = _canonical_signature(resolvent)
                if sig in seen_sigs:
                    continue
                seen_sigs.add(sig)
                step_num += 1
                rid = f"C{len(all_clauses) + 1}"
                all_clauses.append((rid, resolvent, True))
                clause_depth[rid] = new_depth
                st = ResolutionStep(
                    step_num=step_num,
                    resolvent_id=rid,
                    parent1_id=cid_a,
                    parent2_id=cid_b,
                    literal1=l1,
                    literal2=l2,
                    unifier=subst,
                    resolvent=resolvent,
                    depth=new_depth,
                )
                steps.append(st)
                heapq.heappush(pq, (len(resolvent.literals), new_depth, len(all_clauses), rid, resolvent, True))
                if step_num >= max_steps:
                    break
            if step_num >= max_steps:
                break

    reason = (
        f"Limite d'étapes atteinte ({max_steps})"
        if step_num >= max_steps
        else "Espace Set-of-Support saturé sans clause vide"
    )
    return ResolutionResult(
        proved=False,
        query=query,
        initial_clauses=initial_snapshot,
        steps=steps,
        proof_steps=[],
        explored_steps=step_num,
        stop_reason=reason,
    )
