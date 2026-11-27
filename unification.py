"""Moteur d'unification du premier ordre (Bloc 2 - Membre 2).

Avertissement : ce module fait partie d'un prototype pédagogique de simulation
d'aide à la décision aux urgences, sans aucun usage médical réel.

Implémente l'algorithme d'unification de Robinson sans dépendance externe :
- Variables logiques (`Var`), constantes Python (`str`, `int`, `float`, `bool`),
  termes composés (`Compound`) et atomes représentés en tuples `(Predicat, ...)`.
- Vérification d'occurrence (*occurs check*) pour rejeter les termes cycliques.
- Application et composition de substitutions, chaînes de variables et renommage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

Substitution = dict["Var", Any]


@dataclass(frozen=True, order=True)
class Var:
    """Variable logique (ex. `Var("x")`)."""

    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Compound:
    """Terme composé ou atome logique `functor(arg1, ..., argn)`."""

    functor: str
    args: tuple[Any, ...]

    def __init__(self, functor: str, args: Iterable[Any] = ()):
        object.__setattr__(self, "functor", functor)
        object.__setattr__(self, "args", tuple(args))

    def __repr__(self) -> str:
        if not self.args:
            return self.functor
        inner = ", ".join(format_term(a) for a in self.args)
        return f"{self.functor}({inner})"


def is_var(term: Any) -> bool:
    """Retourne True si `term` est une variable logique `Var`."""
    return isinstance(term, Var)


def is_compound(term: Any) -> bool:
    """Retourne True si `term` est un `Compound` ou un tuple non vide (atome M1)."""
    return isinstance(term, Compound) or (isinstance(term, tuple) and len(term) > 0)


def format_term(term: Any) -> str:
    """Formate un terme ou un atome tuple de façon lisible."""
    if isinstance(term, Var):
        return term.name
    if isinstance(term, Compound):
        return repr(term)
    if isinstance(term, tuple) and term:
        args = ",".join(format_term(a) for a in term[1:])
        return f"{term[0]}({args})" if len(term) > 1 else str(term[0])
    return str(term)


def format_subst(subst: Substitution | None) -> str:
    """Formate une substitution `{x/P1, y/Cardiology}` (ou `∅` / `ECHEC`)."""
    if subst is None:
        return "ECHEC"
    if not subst:
        return "{}"
    pairs = ", ".join(f"{v.name}/{format_term(val)}" for v, val in sorted(subst.items(), key=lambda kv: kv[0].name))
    return "{" + pairs + "}"


def _equal_atoms(a: Any, b: Any) -> bool:
    """Compare deux constantes en distinguant booléens et entiers."""
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    return a == b


def walk(term: Any, subst: Substitution) -> Any:
    """Résout une chaîne de variables dans `subst` jusqu'au terme non-lié ou non-variable."""
    seen: set[Var] = set()
    cur = term
    while isinstance(cur, Var) and cur in subst:
        if cur in seen:
            break
        seen.add(cur)
        cur = subst[cur]
    return cur


def occurs_in(var: Var, term: Any, subst: Substitution) -> bool:
    """Vérifie si `var` apparaît dans `term` modulo `subst` (occurs check)."""
    cur = walk(term, subst)
    if cur == var:
        return True
    if isinstance(cur, Compound):
        return any(occurs_in(var, arg, subst) for arg in cur.args)
    if isinstance(cur, tuple):
        return any(occurs_in(var, arg, subst) for arg in cur)
    if isinstance(cur, list):
        return any(occurs_in(var, arg, subst) for arg in cur)
    return False


def apply_subst(term: Any, subst: Substitution, _seen: frozenset[Var] = frozenset()) -> Any:
    """Applique récursivement `subst` à `term` (en résolvant les chaînes de variables)."""
    if not subst:
        return term
    if isinstance(term, Var):
        if term in _seen or term not in subst:
            return term
        return apply_subst(subst[term], subst, _seen | {term})
    if isinstance(term, Compound):
        return Compound(term.functor, tuple(apply_subst(a, subst, _seen) for a in term.args))
    if isinstance(term, tuple):
        return tuple(apply_subst(a, subst, _seen) for a in term)
    if isinstance(term, list):
        return [apply_subst(a, subst, _seen) for a in term]
    return term


def compose_subst(s1: Substitution, s2: Substitution) -> Substitution:
    """Compose deux substitutions (`s1` suivie de `s2`) : `(t · s1) · s2 == t · (s1 ∘ s2)`."""
    composed: Substitution = {v: apply_subst(val, s2) for v, val in s1.items()}
    for v, val in s2.items():
        if v not in composed:
            composed[v] = apply_subst(val, s2)
    return {v: val for v, val in composed.items() if val != v}


def normalize_subst(subst: Substitution) -> Substitution:
    """Résout toutes les chaînes transitives de variables dans une substitution."""
    out: Substitution = {}
    for v, val in subst.items():
        resolved = apply_subst(val, subst, frozenset({v}))
        if resolved != v:
            out[v] = resolved
    return out


def _unify_var(var: Var, x: Any, subst: Substitution, occurs_check: bool) -> Substitution | None:
    """Unifie une variable `var` avec un terme `x` sous la substitution courante."""
    if var in subst:
        return unify(subst[var], x, subst, occurs_check=occurs_check)
    if isinstance(x, Var) and x in subst:
        return unify(var, subst[x], subst, occurs_check=occurs_check)
    if occurs_check and occurs_in(var, x, subst):
        return None
    new_subst = dict(subst)
    new_subst[var] = x
    return normalize_subst(new_subst) if occurs_check else new_subst


def unify(
    x: Any,
    y: Any,
    subst: Substitution | None = None,
    *,
    occurs_check: bool = True,
) -> Substitution | None:
    """Unifie deux termes/atomes `x` et `y` sous la substitution initiale `subst`.

    Supporte :
    - `Var` (variables logiques),
    - `Compound(functor, args)`,
    - tuples non vides `(Predicat, arg1, ...)` (format Membre 1) et listes,
    - constantes (`str`, `int`, `float`, `bool`).

    Retourne une substitution normalisée `dict[Var, Any]` ou `None` en cas d'échec.
    """
    if subst is None:
        subst = {}

    x_resolved = walk(x, subst)
    y_resolved = walk(y, subst)

    if isinstance(x_resolved, Var) and isinstance(y_resolved, Var) and x_resolved == y_resolved:
        return dict(subst)
    if isinstance(x_resolved, Var):
        return _unify_var(x_resolved, y_resolved, subst, occurs_check)
    if isinstance(y_resolved, Var):
        return _unify_var(y_resolved, x_resolved, subst, occurs_check)

    # Interopérabilité Compound <-> tuple atome ("Pred", a1, a2)
    if isinstance(x_resolved, Compound) and isinstance(y_resolved, Compound):
        if x_resolved.functor != y_resolved.functor or len(x_resolved.args) != len(y_resolved.args):
            return None
        return _unify_seq(x_resolved.args, y_resolved.args, subst, occurs_check)

    if isinstance(x_resolved, Compound) and isinstance(y_resolved, tuple) and y_resolved:
        if x_resolved.functor != y_resolved[0] or len(x_resolved.args) != len(y_resolved) - 1:
            return None
        return _unify_seq(x_resolved.args, y_resolved[1:], subst, occurs_check)

    if isinstance(x_resolved, tuple) and x_resolved and isinstance(y_resolved, Compound):
        if x_resolved[0] != y_resolved.functor or len(x_resolved) - 1 != len(y_resolved.args):
            return None
        return _unify_seq(x_resolved[1:], y_resolved.args, subst, occurs_check)

    if isinstance(x_resolved, tuple) and isinstance(y_resolved, tuple):
        if len(x_resolved) != len(y_resolved):
            return None
        return _unify_seq(x_resolved, y_resolved, subst, occurs_check)

    if isinstance(x_resolved, list) and isinstance(y_resolved, list):
        if len(x_resolved) != len(y_resolved):
            return None
        return _unify_seq(x_resolved, y_resolved, subst, occurs_check)

    if _equal_atoms(x_resolved, y_resolved) and not is_compound(x_resolved) and not is_compound(y_resolved):
        return dict(subst)

    return None


def _unify_seq(
    xs: Iterable[Any],
    ys: Iterable[Any],
    subst: Substitution,
    occurs_check: bool,
) -> Substitution | None:
    cur: Substitution | None = dict(subst)
    for a, b in zip(xs, ys):
        if cur is None:
            return None
        cur = unify(a, b, cur, occurs_check=occurs_check)
    return cur


def collect_vars(term: Any) -> set[Var]:
    """Collecte l'ensemble des variables présentes dans `term`."""
    if isinstance(term, Var):
        return {term}
    if isinstance(term, Compound):
        out: set[Var] = set()
        for a in term.args:
            out |= collect_vars(a)
        return out
    if isinstance(term, (tuple, list)):
        out = set()
        for a in term:
            out |= collect_vars(a)
        return out
    return set()


def rename_vars(term: Any, suffix: str | Callable[[str], str] | None = None) -> tuple[Any, Substitution]:
    """Renomme toutes les variables d'un terme pour éviter les collisions (*standardize apart*)."""
    vars_found = sorted(collect_vars(term), key=lambda v: v.name)
    if suffix is None:
        suffix = "_1"
    mapping: Substitution = {}
    for v in vars_found:
        new_name = suffix(v.name) if callable(suffix) else f"{v.name}{suffix}"
        mapping[v] = Var(new_name)
    return apply_subst(term, mapping), mapping


def _split_top_level(text: str) -> list[str]:
    parts, buf, depth = [], [], 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return parts


def parse_term(text: str) -> Any:
    """Parse une chaîne en terme logique.

    Conventions :
    - `?x` ou identifiant commençant par une minuscule (`x`, `p`, `var1`) -> `Var`
    - `f(a, b)` -> `Compound` (si `f` commence par une majuscule et c'est un atome,
      utilisez `parse_atom` pour obtenir un tuple Membre 1)
    - nombre entier ou flottant -> `int` / `float`
    - constante majuscule (`P1`, `Cardiology`) ou entre guillemets -> `str`
    """
    s = text.strip()
    if not s:
        raise ValueError("Terme vide")
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    if "(" in s and s.endswith(")"):
        idx = s.index("(")
        functor = s[:idx].strip()
        inner = s[idx + 1 : -1].strip()
        args = tuple(parse_term(p) for p in _split_top_level(inner)) if inner else ()
        return Compound(functor, args)
    if s.startswith("?"):
        return Var(s[1:])
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    if s[0].islower() or s[0] == "_":
        return Var(s)
    return s


def parse_atom(text: str) -> tuple[Any, ...]:
    """Parse un prédicat texte `RequiresSpecialist(x, Cardiology)` en tuple `(Pred, arg1, ...)`."""
    term = parse_term(text)
    if isinstance(term, Compound):
        return (term.functor, *term.args)
    if isinstance(term, str):
        return (term,)
    raise ValueError(f"Atome invalide : {text}")
