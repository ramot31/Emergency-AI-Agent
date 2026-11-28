"""Tests du contrat Machine Learning du membre 3."""
from __future__ import annotations

from pathlib import Path

import pytest

from environment import Patient
from ml.config import DEFAULT_MODEL_PATH, RISK_LEVELS
from ml.predict_ml import format_ml_risk, ml_predict, predict_risk, to_ml_risk_fact
from ml.preprocessing import make_feature_frame


def _p1() -> Patient:
    return Patient(
        id="P1", age=72, heart_rate=132, systolic_bp=84, spo2=87,
        temperature=39.0, resp_rate=26, symptoms=["ChestPain", "ShortnessOfBreath"],
    )


def test_patient_is_converted_to_stable_features():
    features = make_feature_frame(_p1())
    assert features.loc[0, "symptom_ChestPain"] == 1
    assert features.loc[0, "symptom_ShortnessOfBreath"] == 1
    assert features.loc[0, "symptom_Confusion"] == 0


def test_ml_fact_contract_is_kb_compatible():
    prediction = ("High", 0.87321)
    assert to_ml_risk_fact("P1", prediction) == ("ML_Risk", "P1", "High", 0.8732)
    assert format_ml_risk("P1", prediction) == "ML_Risk(P1,High,0.87)"


def test_hyperparameter_choice_prioritizes_cv_critical_recall():
    from ml.train_ml import select_cv_index
    results = {"params": [{}, {}, {}], "mean_test_critical_recall": [0.7, 0.9, 0.9],
               "mean_test_f1_macro": [0.99, 0.7, 0.8]}
    assert select_cv_index(results) == 2


@pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="Artefact non entraîné : lancer python -m ml.train_ml")
def test_saved_model_loads_and_predicts_a_valid_risk():
    level, probability = predict_risk(_p1())
    assert level in RISK_LEVELS
    assert 0.0 <= probability <= 1.0
    assert to_ml_risk_fact("P1", (level, probability))[0] == "ML_Risk"


@pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="Artefact non entraîné : lancer python -m ml.train_ml")
def test_real_predictor_can_be_injected_into_existing_agent():
    from agent import Agent
    from interface import load_scenario

    state = load_scenario("scenarios/scenario_complet.json")
    agent = Agent(state, ml=ml_predict)
    agent.cycle()
    assert any("ML_Risk(P1," in line for line in agent.journal)


@pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="Artefact non entraîné")
def test_prediction_is_consumed_by_real_knowledge_base():
    from kb import KnowledgeBase
    from ml_dl_bridge import apply_ml_dl_bridge

    level, probability = predict_risk(_p1())
    kb = KnowledgeBase()
    from environment import State
    from facts import state_to_facts
    state = State()
    state.add_patient(_p1())
    kb.assert_facts(state_to_facts(state))
    fact = to_ml_risk_fact("P1", (level, probability))
    kb.assert_fact(fact, source="ml")
    derived, _ = apply_ml_dl_bridge(kb)
    thresholds = {"Critical": (0.8, "CriticalRisk"), "High": (0.75, "HighRisk"),
                  "Medium": (0.6, "ModerateRisk")}
    if level in thresholds and probability >= thresholds[level][0]:
        assert (thresholds[level][1], "P1") in derived
    else:
        assert not derived


@pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="Artefact non entraîné")
def test_cooperation_changes_real_kb_requirements():
    from ml.demo_cooperation import find_demo

    result = find_demo()
    assert result["sans_ml"]["exigences"] != result["avec_ml"]["exigences"]
    assert any("BRIDGE_ML" in line for line in result["avec_ml"]["trace"])
    assert result["prediction"][0] == "ML_Risk"


@pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="Artefact non entraîné")
def test_tree_explanation_matches_inference():
    from ml.explain_ml import explain_prediction

    explanation = explain_prediction(_p1())
    level, probability = predict_risk(_p1())
    assert explanation["prediction"] == level
    assert round(explanation["score"], 4) == probability
    assert explanation["chemin"]
    assert all(step["condition"] in {"<=", ">"} for step in explanation["chemin"])


def test_diagnostics_identify_missing_outliers_and_unknown_symptoms():
    from ml.predict_ml import inspect_patient

    patient = vars(_p1()).copy()
    patient["heart_rate"] = None
    patient["spo2"] = 150
    patient["symptoms"] = ["ChestPain", "SymptomeInconnu"]
    messages = inspect_patient(patient)
    assert any("heart_rate manquant" in message for message in messages)
    assert any("spo2 hors" in message for message in messages)
    assert any("SymptomeInconnu" in message for message in messages)


@pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="Artefact non entraîné")
def test_diagnostic_interface_keeps_original_prediction_contract():
    from ml.predict_ml import predict_risk_with_diagnostics

    result = predict_risk_with_diagnostics(_p1())
    assert (result["level"], result["score"]) == predict_risk(_p1())
    assert result["warnings"] == []


def test_diagnostic_interface_rejects_missing_required_key():
    from ml.predict_ml import predict_risk_with_diagnostics

    with pytest.raises(ValueError, match="Champs requis absents"):
        predict_risk_with_diagnostics({"age": 45})
