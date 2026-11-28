"""Pont ML / DL vers la Base de Connaissances (Bloc 2 + intégration Bloc 6 - Membre 2).

Avertissement : ce module fait partie d'un prototype pédagogique de simulation
d'aide à la décision aux urgences, sans aucun usage médical réel.

Convertit les faits d'apprentissage `ML_Risk(p, niveau, prob)` (Membre 3) et
`DL_Risk(p, niveau, prob)` (Membre 4) en faits symboliques exploitables par le
moteur logique (`CriticalRisk`, `HighRisk`, `ModerateRisk`, `RequiresUrgentEvaluation`),
avec seuils de confiance paramétrables et résolution tracée des conflits ML/DL.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from kb import Justification, KnowledgeBase

LEVEL_RANK: dict[str, int] = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}

LEVEL_TO_PREDICATE: dict[str, str] = {
    "Critical": "CriticalRisk",
    "High": "HighRisk",
    "Medium": "ModerateRisk",
}


@dataclass(frozen=True)
class BridgeThresholds:
    """Seuils de confiance paramétrables et justifiés pour le pont ML/DL -> KB.

    - `dl_critical_threshold = 0.80` : un risque critique issu d'une série temporelle
      fiable (>= 80 %) déclenche `CriticalRisk` et l'orientation USI.
    - `ml_critical_threshold = 0.80` : seuil élevé pour éviter les faux positifs USI
      à partir d'un simple instantané.
    - `dl_high_threshold = 0.75` / `ml_high_threshold = 0.75` : seuil d'alerte élevée.
    - `medium_threshold = 0.60` : seuil modéré.
    - `ml_chest_pain_threshold = 0.70` : combiné à `ChestPain`, déclenche une
      évaluation urgente même avec une confiance ML intermédiaire.
    - `dl_priority_threshold = 0.80` : au-delà de ce seuil, le DL prime lors d'une
      divergence ML/DL car il capte l'évolution temporelle du patient.
    - `ml_safety_override_threshold = 0.85` : garde-fou de précaution si ML prédit un
      niveau supérieur au DL avec une très forte confiance (>= 85 %).
    """

    dl_critical_threshold: float = 0.80
    ml_critical_threshold: float = 0.80
    dl_high_threshold: float = 0.75
    ml_high_threshold: float = 0.75
    medium_threshold: float = 0.60
    ml_chest_pain_threshold: float = 0.70
    dl_priority_threshold: float = 0.80
    ml_safety_override_threshold: float = 0.85
    min_precaution_prob: float = 0.50


@dataclass(frozen=True)
class ConflictDecision:
    """Décision explicable lors d'une divergence entre ML_Risk et DL_Risk."""

    patient_id: str
    ml_fact: tuple[str, str, str, float]
    dl_fact: tuple[str, str, str, float]
    chosen_source: str  # "DL" | "ML" | "PRECAUTION_MAX"
    chosen_level: str
    chosen_prob: float
    reason: str


def _passes_threshold(source: str, level: str, prob: float, t: BridgeThresholds) -> bool:
    if level == "Critical":
        return prob >= (t.dl_critical_threshold if source == "DL" else t.ml_critical_threshold)
    if level == "High":
        return prob >= (t.dl_high_threshold if source == "DL" else t.ml_high_threshold)
    if level == "Medium":
        return prob >= t.medium_threshold
    return False


def moderate_ml_only_level(ml_level: str, rule_level: str | None) -> str:
    """ML High seul devient Medium provisoire, sans abaisser Critical ou les règles."""
    if ml_level == "High" and rule_level is None:
        return "Medium"
    return ml_level


def resolve_ml_dl_conflict(
    ml_fact: tuple[str, str, str, float],
    dl_fact: tuple[str, str, str, float],
    thresholds: BridgeThresholds,
) -> ConflictDecision:
    """Résout une divergence entre `ML_Risk(p, l1, p1)` et `DL_Risk(p, l2, p2)`."""
    _, pid, ml_lvl, ml_p = ml_fact
    _, _, dl_lvl, dl_p = dl_fact
    ml_rank = LEVEL_RANK.get(ml_lvl, 0)
    dl_rank = LEVEL_RANK.get(dl_lvl, 0)

    if dl_p >= thresholds.dl_priority_threshold:
        if dl_rank >= ml_rank:
            return ConflictDecision(
                patient_id=pid,
                ml_fact=ml_fact,
                dl_fact=dl_fact,
                chosen_source="DL",
                chosen_level=dl_lvl,
                chosen_prob=dl_p,
                reason=(
                    f"Conflit ML={ml_lvl}({ml_p:.2f}) vs DL={dl_lvl}({dl_p:.2f}) : "
                    f"DL prime ({dl_p:.2f} >= {thresholds.dl_priority_threshold:.2f}, analyse temporelle)"
                ),
            )
        if ml_p >= thresholds.ml_safety_override_threshold:
            return ConflictDecision(
                patient_id=pid,
                ml_fact=ml_fact,
                dl_fact=dl_fact,
                chosen_source="PRECAUTION_MAX",
                chosen_level=ml_lvl,
                chosen_prob=ml_p,
                reason=(
                    f"Conflit ML={ml_lvl}({ml_p:.2f}) vs DL={dl_lvl}({dl_p:.2f}) : "
                    f"principe de précaution (ML >= {thresholds.ml_safety_override_threshold:.2f} sur niveau supérieur)"
                ),
            )
        return ConflictDecision(
            patient_id=pid,
            ml_fact=ml_fact,
            dl_fact=dl_fact,
            chosen_source="DL",
            chosen_level=dl_lvl,
            chosen_prob=dl_p,
            reason=(
                f"Conflit ML={ml_lvl}({ml_p:.2f}) vs DL={dl_lvl}({dl_p:.2f}) : "
                f"DL prime ({dl_p:.2f} >= {thresholds.dl_priority_threshold:.2f}, ML sous garde-fou)"
            ),
        )

    # Sinon : DL sous le seuil -> on retient le niveau le plus élevé par précaution
    if ml_rank >= dl_rank and ml_p >= thresholds.min_precaution_prob:
        chosen_src, chosen_lvl, chosen_p = "PRECAUTION_MAX", ml_lvl, ml_p
    elif dl_p >= thresholds.min_precaution_prob:
        chosen_src, chosen_lvl, chosen_p = "PRECAUTION_MAX", dl_lvl, dl_p
    else:
        chosen_src, chosen_lvl, chosen_p = (
            ("ML", ml_lvl, ml_p) if ml_rank >= dl_rank else ("DL", dl_lvl, dl_p)
        )
    return ConflictDecision(
        patient_id=pid,
        ml_fact=ml_fact,
        dl_fact=dl_fact,
        chosen_source=chosen_src,
        chosen_level=chosen_lvl,
        chosen_prob=chosen_p,
        reason=(
            f"Conflit ML={ml_lvl}({ml_p:.2f}) vs DL={dl_lvl}({dl_p:.2f}) : "
            f"DL sous seuil ({thresholds.dl_priority_threshold:.2f}), niveau le plus élevé retenu par précaution ({chosen_lvl})"
        ),
    )


def apply_ml_dl_bridge(
    kb: KnowledgeBase,
    thresholds: BridgeThresholds | None = None,
) -> tuple[set[tuple[Any, ...]], list[str]]:
    """Transforme les faits `ML_Risk` et `DL_Risk` présents dans `kb` en faits dérivés tracés."""
    t = thresholds or BridgeThresholds()
    trace: list[str] = []
    derived: set[tuple[Any, ...]] = set()

    ml_by_patient: dict[str, tuple[str, str, str, float]] = {}
    dl_by_patient: dict[str, tuple[str, str, str, float]] = {}

    for f in kb.all_facts_sorted():
        if len(f) == 4 and f[0] == "ML_Risk":
            if (f[2] not in LEVEL_RANK or not isinstance(f[3], (int, float))
                    or isinstance(f[3], bool) or not math.isfinite(f[3]) or not 0 <= f[3] <= 1):
                trace.append(f"[BRIDGE_ML_INVALID] Prédiction ignorée : {f}")
                continue
            ml_by_patient[str(f[1])] = f  # type: ignore[assignment]
        elif len(f) == 4 and f[0] == "DL_Risk":
            if (f[2] not in LEVEL_RANK or not isinstance(f[3], (int, float))
                    or isinstance(f[3], bool) or not math.isfinite(f[3]) or not 0 <= f[3] <= 1):
                trace.append(f"[BRIDGE_DL_INVALID] Prédiction ignorée : {f}")
                continue
            dl_by_patient[str(f[1])] = f  # type: ignore[assignment]

    patients = sorted(set(ml_by_patient) | set(dl_by_patient))
    rule_facts = set()
    if any(pid not in dl_by_patient for pid in ml_by_patient):
        from inference import forward_chain

        rule_kb = KnowledgeBase(rules=kb.rules)
        rule_kb.assert_facts(f for f in kb.base_facts if f[0] not in {"ML_Risk", "DL_Risk"})
        forward_chain(rule_kb)
        rule_facts = rule_kb.facts
    for pid in patients:
        ml_fact = ml_by_patient.get(pid)
        dl_fact = dl_by_patient.get(pid)

        # Cas 1 : ML et DL sont tous deux présents
        if ml_fact is not None and dl_fact is not None:
            _, _, ml_lvl, ml_p = ml_fact
            _, _, dl_lvl, dl_p = dl_fact

            if ml_lvl != dl_lvl:
                decision = resolve_ml_dl_conflict(ml_fact, dl_fact, t)
                conflict_fact = ("ResourceConflictMLDL", pid)
                j_conf = Justification(
                    rule_id="BRIDGE_CONFLICT",
                    premises=(ml_fact, dl_fact),
                    substitution_str=f"{{p/{pid}}}",
                    note=decision.reason,
                )
                if kb.add_derived_fact(conflict_fact, j_conf):
                    derived.add(conflict_fact)
                trace.append(f"[BRIDGE_CONFLICT] {decision.reason}")

                pred = LEVEL_TO_PREDICATE.get(decision.chosen_level)
                eff_src = "DL" if decision.chosen_source == "DL" else "ML"
                if pred and _passes_threshold(eff_src, decision.chosen_level, decision.chosen_prob, t):
                    out_fact = (pred, pid)
                    j = Justification(
                        rule_id="BRIDGE_ML_DL",
                        premises=(ml_fact, dl_fact),
                        substitution_str=f"{{p/{pid}}}",
                        note=decision.chosen_source,
                    )
                    if kb.add_derived_fact(out_fact, j):
                        derived.add(out_fact)
                        trace.append(j.format_line(out_fact))
            else:
                # Accord ML et DL
                pred = LEVEL_TO_PREDICATE.get(dl_lvl)
                if pred and (
                    _passes_threshold("DL", dl_lvl, dl_p, t)
                    or _passes_threshold("ML", ml_lvl, ml_p, t)
                ):
                    out_fact = (pred, pid)
                    j = Justification(
                        rule_id="BRIDGE_AGREE",
                        premises=(ml_fact, dl_fact),
                        substitution_str=f"{{p/{pid}}}",
                        note="accord ML/DL",
                    )
                    if kb.add_derived_fact(out_fact, j):
                        derived.add(out_fact)
                        trace.append(j.format_line(out_fact))

        # Cas 2 : un seul modèle disponible (ex. DL absent si historique < 3 points)
        else:
            src_name, fact = ("DL", dl_fact) if dl_fact is not None else ("ML", ml_fact)
            assert fact is not None
            _, _, lvl, prob = fact
            accepted = _passes_threshold(src_name, lvl, prob, t)
            rule_level = next((name for name in ("Critical", "High", "Medium")
                               if (LEVEL_TO_PREDICATE[name], pid) in rule_facts), None)
            effective_level = moderate_ml_only_level(lvl, rule_level) if src_name == "ML" and accepted else lvl
            pred = LEVEL_TO_PREDICATE.get(effective_level)
            if pred and accepted:
                out_fact = (pred, pid)
                j = Justification(
                    rule_id=f"BRIDGE_{src_name}",
                    premises=(fact,),
                    substitution_str=f"{{p/{pid}}}",
                    note=(f"score={prob:.2f}" + (" ; High ML sans corroboration logique : Medium provisoire"
                                                 if effective_level != lvl else "")),
                )
                if kb.add_derived_fact(out_fact, j):
                    derived.add(out_fact)
                    trace.append(j.format_line(out_fact))

        # Règle combinée : ML High + ChestPain -> RequiresUrgentEvaluation
        if ml_fact is not None:
            _, _, ml_lvl, ml_p = ml_fact
            chest_fact = ("HasSymptom", pid, "ChestPain")
            if ml_lvl in ("High", "Critical") and ml_p >= t.ml_chest_pain_threshold and kb.contains(chest_fact):
                urg_fact = ("RequiresUrgentEvaluation", pid)
                j_urg = Justification(
                    rule_id="BRIDGE_ML_CHESTPAIN",
                    premises=(ml_fact, chest_fact),
                    substitution_str=f"{{p/{pid}}}",
                    note=f"ML={ml_lvl}({ml_p:.2f}) + ChestPain",
                )
                if kb.add_derived_fact(urg_fact, j_urg):
                    derived.add(urg_fact)
                    trace.append(j_urg.format_line(urg_fact))

    kb.trace_log.extend(trace)
    return derived, trace
