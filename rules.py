"""Base de règles logiques du domaine des urgences (Bloc 2 - Membre 2).

Avertissement : ce module fait partie d'un prototype pédagogique de simulation
d'aide à la décision aux urgences, sans aucun usage médical réel.

Contient les clauses de Horn avec variables utilisées par le moteur d'inférence
(chaînage avant/arrière) et le moteur de résolution par réfutation.
Réutilise strictement les prédicats discrétisés du Membre 1 (`LowBloodPressure`,
`HighBloodPressure`, `LowOxygen`, `Tachycardia`, `Fever`, `HasSymptom`, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from unification import Substitution, Var, apply_subst, format_term

# Variables logiques partagées pour la définition lisible des règles
X = Var("x")
A = Var("a")
R = Var("r")
S = Var("s")


@dataclass(frozen=True)
class Guard:
    """Contrainte arithmétique ou relationnelle évaluée après unification des prémisses."""

    label: str
    predicate: Callable[[Substitution], bool]

    def holds(self, subst: Substitution) -> bool:
        return bool(self.predicate(subst))


@dataclass(frozen=True)
class Rule:
    """Règle logique sous forme de clause de Horn : P1 ∧ ... ∧ Pn [∧ Guards] -> Conclusion."""

    rule_id: str
    premises: tuple[tuple[Any, ...], ...]
    conclusion: tuple[Any, ...]
    guards: tuple[Guard, ...] = ()
    description: str = ""

    @property
    def is_pure_horn(self) -> bool:
        """Retourne True si la règle ne contient pas de garde procédurale (compatible FNC directe)."""
        return len(self.guards) == 0

    def format_rule(self) -> str:
        """Représentation lisible `P1 ^ P2 -> Conclusion`."""
        prem = " ^ ".join(format_term(p) for p in self.premises)
        if self.guards:
            prem += " ^ " + " ^ ".join(f"[{g.label}]" for g in self.guards)
        return f"[{self.rule_id}] {prem} -> {format_term(self.conclusion)}"


def _num_ge(var: Var, threshold: float) -> Guard:
    return Guard(
        label=f"{var.name} >= {threshold}",
        predicate=lambda sub: isinstance(apply_subst(var, sub), (int, float))
        and not isinstance(apply_subst(var, sub), bool)
        and apply_subst(var, sub) >= threshold,
    )


def _num_gt(var: Var, threshold: float) -> Guard:
    return Guard(
        label=f"{var.name} > {threshold}",
        predicate=lambda sub: isinstance(apply_subst(var, sub), (int, float))
        and not isinstance(apply_subst(var, sub), bool)
        and apply_subst(var, sub) > threshold,
    )


def _num_lt(var: Var, threshold: float) -> Guard:
    return Guard(
        label=f"{var.name} < {threshold}",
        predicate=lambda sub: isinstance(apply_subst(var, sub), (int, float))
        and not isinstance(apply_subst(var, sub), bool)
        and apply_subst(var, sub) < threshold,
    )


def build_default_rules() -> list[Rule]:
    """Construit l'ensemble des règles logiques du Bloc 2 (clauses de Horn avec variables)."""
    return [
        # --- 1. Discrétisation complémentaire sur mesures non discrétisées par Membre 1 ---
        Rule(
            "R01",
            (("Age", X, A),),
            ("Elderly", X),
            guards=(_num_ge(A, 65),),
            description="Patient âgé (>= 65 ans)",
        ),
        Rule(
            "R02",
            (("RespiratoryRate", X, R),),
            ("Tachypnea", X),
            guards=(_num_gt(R, 22),),
            description="Fréquence respiratoire élevée (> 22/min)",
        ),
        Rule(
            "R03",
            (("OxygenSaturation", X, S),),
            ("SevereHypoxia", X),
            guards=(_num_lt(S, 90),),
            description="Hypoxie sévère (SpO2 < 90 %)",
        ),
        # --- 2. Règles de risque élevé (HighRisk) ---
        Rule(
            "R04",
            (("HasSymptom", X, "ChestPain"), ("LowBloodPressure", X)),
            ("HighRisk", X),
            description="Douleur thoracique + hypotension -> risque élevé",
        ),
        Rule(
            "R05",
            (("LowOxygen", X), ("Tachycardia", X)),
            ("HighRisk", X),
            description="Hypoxie + tachycardie -> risque élevé",
        ),
        Rule(
            "R06",
            (("Elderly", X), ("Fever", X), ("LowOxygen", X)),
            ("HighRisk", X),
            description="Patient âgé fébrile hypoxique -> risque élevé",
        ),
        Rule(
            "R07",
            (("HasSymptom", X, "ShortnessOfBreath"), ("Tachypnea", X)),
            ("HighRisk", X),
            description="Dyspnée + tachypnée -> risque élevé",
        ),
        Rule(
            "R08",
            (("HasSymptom", X, "ChestPain"), ("HighBloodPressure", X)),
            ("HighRisk", X),
            description="Douleur thoracique + hypertension -> risque élevé",
        ),
        # --- 3. Règles de risque critique (CriticalRisk) ---
        Rule(
            "R09",
            (("LowBloodPressure", X), ("SevereHypoxia", X)),
            ("CriticalRisk", X),
            description="Hypotension + hypoxie sévère (SpO2 < 90) -> critique",
        ),
        Rule(
            "R10",
            (("HasSymptom", X, "ChestPain"), ("LowOxygen", X), ("Elderly", X)),
            ("CriticalRisk", X),
            description="Douleur thoracique + hypoxie chez patient âgé -> critique",
        ),
        Rule(
            "R11",
            (("HasSymptom", X, "Confusion"), ("LowBloodPressure", X)),
            ("CriticalRisk", X),
            description="Confusion + hypotension -> critique",
        ),
        Rule(
            "R12",
            (("CriticalRisk", X),),
            ("HighRisk", X),
            description="Tout risque critique implique un risque élevé",
        ),
        # --- 4. Règle de risque modéré ---
        Rule(
            "R13",
            (("Patient", X), ("Fever", X)),
            ("ModerateRisk", X),
            description="Patient fébrile -> au moins risque modéré",
        ),
        # --- 5. Prise en charge urgente, USI et surveillance ---
        Rule(
            "R14",
            (("HighRisk", X),),
            ("RequiresUrgentEvaluation", X),
            description="Risque élevé -> évaluation urgente requise",
        ),
        Rule(
            "R15",
            (("CriticalRisk", X),),
            ("RequiresICU", X),
            description="Risque critique -> lit de soins intensifs (USI) requis",
        ),
        Rule(
            "R16",
            (("CriticalRisk", X),),
            ("MonitoringRequired", X),
            description="Risque critique -> surveillance continue requise",
        ),
        # --- 6. Spécialistes requis ---
        Rule(
            "R17",
            (("HighRisk", X), ("HasSymptom", X, "ChestPain")),
            ("RequiresSpecialist", X, "Cardiology"),
            description="Risque élevé avec douleur thoracique -> cardiologue",
        ),
        Rule(
            "R18",
            (("HasSymptom", X, "ShortnessOfBreath"), ("LowOxygen", X)),
            ("RequiresSpecialist", X, "Pulmonology"),
            description="Dyspnée + hypoxie -> pneumologue",
        ),
        Rule(
            "R19",
            (("Patient", X), ("HasSymptom", X, "AbdominalPain")),
            ("RequiresSpecialist", X, "General"),
            description="Douleur abdominale -> médecin généraliste",
        ),
        Rule(
            "R20",
            (("ModerateRisk", X), ("Fever", X)),
            ("RequiresSpecialist", X, "General"),
            description="Fièvre modérée -> médecin généraliste",
        ),
        Rule(
            "R21",
            (("Patient", X), ("HasSymptom", X, "Confusion")),
            ("RequiresSpecialist", X, "Neurology"),
            description="Confusion -> neurologue",
        ),
        # --- 7. Lits d'urgence et salles d'examen ---
        Rule(
            "R22",
            (("HighRisk", X),),
            ("RequiresEmergencyBed", X),
            description="Risque élevé -> lit d'urgence requis (sauf si USI)",
        ),
        Rule(
            "R23",
            (("ModerateRisk", X),),
            ("RequiresEmergencyBed", X),
            description="Risque modéré -> lit d'urgence requis",
        ),
        Rule(
            "R24",
            (("PatientWaiting", X),),
            ("RequiresExamRoom", X),
            description="Patient en attente -> salle d'examen requise",
        ),
        # --- 8. Priorités (1 = plus critique, 4 = standard) ---
        Rule(
            "R25",
            (("CriticalRisk", X),),
            ("Priority", X, 1),
            description="Critique -> priorité 1",
        ),
        Rule(
            "R26",
            (("HighRisk", X),),
            ("Priority", X, 2),
            description="Risque élevé -> priorité 2",
        ),
        Rule(
            "R27",
            (("ModerateRisk", X),),
            ("Priority", X, 3),
            description="Risque modéré -> priorité 3",
        ),
        Rule(
            "R28",
            (("Patient", X),),
            ("Priority", X, 4),
            description="Patient standard -> priorité de base 4",
        ),
    ]


DEFAULT_RULES: tuple[Rule, ...] = tuple(build_default_rules())
