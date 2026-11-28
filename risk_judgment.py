"""Journalise la politique de précaution entre règles et prédiction ML.

Simulation académique uniquement. Les seuils et scores ne constituent pas des
garanties médicales. Le jugement n'ajoute aucune conclusion à la KB : il rend
explicite la priorité que le pont et les règles existants ont effectivement produite.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ml_dl_bridge import BridgeThresholds, LEVEL_RANK, _passes_threshold

if TYPE_CHECKING:
    from kb import KnowledgeBase

RISK_PREDICATES = (("CriticalRisk", "Critical"), ("HighRisk", "High"),
                   ("ModerateRisk", "Medium"))


def _highest_risk(facts: set[tuple], pid: str) -> str | None:
    return next((level for predicate, level in RISK_PREDICATES if (predicate, pid) in facts), None)


def trace_rule_ml_judgment(kb: KnowledgeBase) -> list[str]:
    """Compare au même instant les règles seules et le résultat effectif intégré.

    Les conclusions des règles sont des bornes minimales, pas des classes
    exclusives. Le DL conserve sa politique de conflit existante et peut influer
    sur le résultat final ; le journal ne prétend pas résoudre ce conflit ici.
    """
    from inference import forward_chain
    from kb import KnowledgeBase

    predictions = [f for f in kb.base_facts if len(f) == 4 and f[0] == "ML_Risk"]
    if not predictions:
        return []
    baseline = KnowledgeBase(rules=kb.rules)
    baseline.assert_facts(f for f in kb.base_facts if f[0] not in {"ML_Risk", "DL_Risk"})
    forward_chain(baseline)
    lines = []
    for fact in sorted(predictions, key=str):
        _, pid, level, score = fact
        minimum = _highest_risk(baseline.facts, pid)
        effective = _highest_risk(kb.facts, pid)
        minimum_text = minimum or "aucune conclusion de risque"
        effective_text = effective or "aucune conclusion de risque (priorité de base)"
        valid = (level in LEVEL_RANK and isinstance(score, (int, float))
                 and not isinstance(score, bool) and math.isfinite(score) and 0 <= score <= 1)
        if not valid:
            status = "prédiction invalide ; vérification nécessaire"
        elif level != "Low" and not _passes_threshold("ML", level, score, BridgeThresholds()):
            status = "score sous le seuil principal ML ; d'autres règles combinées peuvent encore s'appliquer"
        elif minimum is None:
            status = "indication ML seule ; absence de conclusion des règles ne signifie pas Low"
        elif LEVEL_RANK[level] < LEVEL_RANK[minimum]:
            status = "désaccord : ML sous le minimum des règles ; alerte des règles conservée, réévaluation recommandée"
        elif LEVEL_RANK[level] == LEVEL_RANK[minimum]:
            status = "accord sur le niveau minimum"
        else:
            status = "ML supérieur au minimum des règles : complément, pas contradiction automatique"
        lines.append(f"[JUGEMENT_REGLES_ML] {pid} : règles={minimum_text} ; ML={level}({score}) ; "
                     f"{status} ; risque effectivement retenu={effective_text}. "
                     "Score non validé comme probabilité médicale.")
    return lines
