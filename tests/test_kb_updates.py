"""Tests unitaires de mise à jour dynamique et maintien de cohérence (Bloc 2 - Membre 2)."""
from facts import state_to_facts
from interface import load_scenario
from kb import KnowledgeBase


def test_update_icu_unavailable_and_available():
    kb = KnowledgeBase()
    kb.load_state_facts(state_to_facts(load_scenario("scenarios/scenario_complet.json")))
    kb.recompute()
    assert kb.contains(("Available", "ICU1"))

    # Contrat PDF (icu_unavailable)
    res1 = kb.update({"type": "icu_unavailable", "id": "ICU1"})
    assert ("Available", "ICU1") in res1["retracted_or_invalidated"]
    assert not kb.contains(("Available", "ICU1"))
    # P1 requiert toujours l'USI mais ICU1 n'est plus disponible
    assert kb.contains(("RequiresICU", "P1"))

    # Retour disponible
    kb.update({"type": "resource_available", "id": "ICU1"})
    assert kb.contains(("Available", "ICU1"))


def test_update_new_patient_and_deterioration_vs_recovery():
    kb = KnowledgeBase()
    kb.load_state_facts(state_to_facts(load_scenario("scenarios/scenario_complet.json")))
    kb.recompute()

    # Arrivée P4 critique (scenarios/event_nouveau_patient_critique.json)
    kb.update(
        {
            "type": "new_patient",
            "patient": {
                "id": "P4",
                "age": 65,
                "heart_rate": 140,
                "systolic_bp": 80,
                "spo2": 85,
                "temperature": 38.5,
                "resp_rate": 30,
                "symptoms": ["ChestPain", "Confusion"],
            },
        }
    )
    assert kb.contains(("CriticalRisk", "P4"))
    assert kb.contains(("RequiresICU", "P4"))
    assert kb.contains(("RequiresSpecialist", "P4", "Cardiology"))
    assert kb.contains(("RequiresSpecialist", "P4", "Neurology"))

    # P2 se dégrade brutalement
    assert not kb.contains(("CriticalRisk", "P2"))
    kb.update({"type": "new_measurement", "patient": "P2", "hr": 135, "bp": 82, "spo2": 86})
    assert kb.contains(("CriticalRisk", "P2"))
    assert kb.contains(("RequiresICU", "P2"))

    # P2 se stabilise -> le maintien de cohérence invalide CriticalRisk(P2) et RequiresICU(P2) !
    res_rec = kb.update({"type": "new_measurement", "patient": "P2", "hr": 82, "bp": 120, "spo2": 98})
    assert ("CriticalRisk", "P2") in res_rec["retracted_or_invalidated"]
    assert ("RequiresICU", "P2") in res_rec["retracted_or_invalidated"]
    assert not kb.contains(("CriticalRisk", "P2"))
    assert not kb.contains(("RequiresICU", "P2"))
    # Mais P2 a toujours Fever -> reste ModerateRisk
    assert kb.contains(("ModerateRisk", "P2"))


def test_update_dl_prediction_invalidates_obsolete_critical():
    kb = KnowledgeBase()
    kb.assert_fact(("Patient", "P9"))
    kb.assert_fact(("ML_Risk", "P9", "Medium", 0.65), source="ml")
    kb.assert_fact(("DL_Risk", "P9", "Critical", 0.91), source="dl")
    kb.recompute()
    assert kb.contains(("CriticalRisk", "P9"))
    assert kb.contains(("RequiresICU", "P9"))

    # Nouvelle prédiction DL : le patient descend à Medium (fiable) -> CriticalRisk/RequiresICU invalidés
    res = kb.update({"type": "dl_prediction", "patient": "P9", "level": "Medium", "prob": 0.88})
    assert ("CriticalRisk", "P9") in res["retracted_or_invalidated"]
    assert ("RequiresICU", "P9") in res["retracted_or_invalidated"]
    assert not kb.contains(("CriticalRisk", "P9"))
    assert not kb.contains(("RequiresICU", "P9"))
    assert kb.contains(("ModerateRisk", "P9"))

    # Si DL passe à Low avec forte confiance (0.90 >= 0.80), DL prime et ModerateRisk disparaît aussi
    res2 = kb.update({"type": "dl_prediction", "patient": "P9", "level": "Low", "prob": 0.90})
    assert ("ModerateRisk", "P9") in res2["retracted_or_invalidated"]
    assert not kb.contains(("ModerateRisk", "P9"))
