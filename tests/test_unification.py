"""Tests unitaires pour le module d'unification (Bloc 2 - Membre 2)."""
from unification import (
    Compound,
    Var,
    apply_subst,
    compose_subst,
    format_subst,
    parse_atom,
    parse_term,
    rename_vars,
    unify,
)


def test_unify_specialist_example():
    x = Var("x")
    pat = ("RequiresSpecialist", x, "Cardiology")
    fact = ("RequiresSpecialist", "P1", "Cardiology")
    subst = unify(pat, fact)
    assert subst == {x: "P1"}
    assert apply_subst(pat, subst) == fact
    assert format_subst(subst) == "{x/P1}"


def test_unify_from_parsed_strings():
    a1 = parse_atom("RequiresSpecialist(x, Cardiology)")
    a2 = parse_atom("RequiresSpecialist(P1, Cardiology)")
    subst = unify(a1, a2)
    assert subst == {Var("x"): "P1"}


def test_unify_compound_terms_and_tuple_interop():
    x, y = Var("x"), Var("y")
    t1 = Compound("pair", (x, Compound("score", ("High", y))))
    t2 = Compound("pair", ("P1", Compound("score", ("High", 0.87))))
    assert unify(t1, t2) == {x: "P1", y: 0.87}

    # Interopérabilité Compound <-> tuple Membre 1
    assert unify(Compound("RequiresICU", (x,)), ("RequiresICU", "P1")) == {x: "P1"}


def test_unify_variable_chains():
    x, y, z = Var("x"), Var("y"), Var("z")
    # x = y, y = z, z = P1 -> {x: P1, y: P1, z: P1}
    s = unify((x, y, z), (y, z, "P1"))
    assert s == {x: "P1", y: "P1", z: "P1"}
    assert apply_subst((x, y, z), s) == ("P1", "P1", "P1")


def test_unify_failures():
    x = Var("x")
    # Prédicats différents
    assert unify(("HighRisk", x), ("CriticalRisk", "P1")) is None
    # Arité différente
    assert unify(("Age", x), ("Age", "P1", 72)) is None
    # Constantes incompatibles
    assert unify(("RequiresSpecialist", x, "Cardiology"), ("RequiresSpecialist", "P1", "Pulmonology")) is None
    # Même variable liée à deux constantes différentes
    assert unify((x, x), ("P1", "P2")) is None
    # Bool vs int ne doivent pas s'unifier
    assert unify(True, 1) is None


def test_occurs_check_prevents_cycles():
    x, y = Var("x"), Var("y")
    # x = f(x) doit échouer
    assert unify(x, Compound("f", (x,))) is None
    # x = y et y = f(x) doit échouer (cycle indirect)
    assert unify((x, y), (y, Compound("f", (x,)))) is None
    # Sans occurs check, l'unification accepte le lien (usage interne contrôlé)
    assert unify(x, Compound("f", (x,)), occurs_check=False) is not None


def test_compose_subst():
    x, y = Var("x"), Var("y")
    s1 = {x: Compound("f", (y,))}
    s2 = {y: "P1"}
    comp = compose_subst(s1, s2)
    assert comp == {x: Compound("f", ("P1",)), y: "P1"}
    term = (x, y)
    assert apply_subst(apply_subst(term, s1), s2) == apply_subst(term, comp)


def test_rename_vars_and_parse_term():
    x, y = Var("x"), Var("y")
    term = ("HasSymptom", x, y)
    renamed, mapping = rename_vars(term, "_r1")
    assert renamed == ("HasSymptom", Var("x_r1"), Var("y_r1"))
    assert mapping == {x: Var("x_r1"), y: Var("y_r1")}
    assert isinstance(parse_term("72"), int)
    assert isinstance(parse_term("0.91"), float)
