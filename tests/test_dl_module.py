"""Tests unitaires et d'intégration du module Deep Learning (Membre 4)."""
from __future__ import annotations

import unittest

from agent import Agent
from dl.config import DEFAULT_DATASET_PATH, RISK_LEVELS
from dl.data_generation import load_sequences_from_csv
from dl.predict_dl import (
    dl_predict,
    format_dl_risk,
    predict_deterioration,
    to_dl_risk_fact,
    update_and_repredict,
)
from dl.preprocessing import split_sequences_by_patient
from interface import load_scenario
from kb import KnowledgeBase
from ml_dl_bridge import apply_ml_dl_bridge


class TestDLModule(unittest.TestCase):
    def test_short_sequence_returns_none(self) -> None:
        self.assertIsNone(dl_predict([]))
        self.assertIsNone(dl_predict([{"hr": 80, "bp": 120, "spo2": 98}]))
        self.assertIsNone(dl_predict([{"hr": 80, "bp": 120, "spo2": 98}, {"hr": 82, "bp": 118, "spo2": 97}]))

    def test_p1_scenario_predicts_critical(self) -> None:
        state = load_scenario("scenarios/scenario_complet.json")
        p1 = state.patients["P1"]
        pred = predict_deterioration(p1.history)
        self.assertIsNotNone(pred)
        level, prob = pred
        self.assertEqual(level, "Critical")
        self.assertGreaterEqual(prob, 0.80)
        self.assertLessEqual(prob, 1.0)

        fact = to_dl_risk_fact("P1", pred)
        self.assertEqual(fact[0], "DL_Risk")
        self.assertEqual(fact[1], "P1")
        self.assertEqual(fact[2], "Critical")
        self.assertTrue(format_dl_risk("P1", pred).startswith("DL_Risk(P1,Critical,"))

    def test_p2_scenario_returns_none(self) -> None:
        state = load_scenario("scenarios/scenario_complet.json")
        p2 = state.patients["P2"]
        self.assertIsNone(dl_predict(p2.history))

    def test_dynamic_reinference_changes_prediction(self) -> None:
        history = [
            {"hr": 98, "bp": 112, "spo2": 95},
            {"hr": 102, "bp": 106, "spo2": 94},
            {"hr": 108, "bp": 100, "spo2": 93},
        ]
        res = update_and_repredict(history, {"hr": 126, "bp": 86, "spo2": 88})
        self.assertEqual(len(res["history"]), 4)
        self.assertIn(res["prediction"][0], RISK_LEVELS)
        self.assertTrue(res["changed"])

    def test_no_patient_leakage_in_split(self) -> None:
        seqs = load_sequences_from_csv(DEFAULT_DATASET_PATH)
        train_s, val_s, test_s = split_sequences_by_patient(seqs)
        train_ids = {s["patient_id"] for s in train_s}
        val_ids = {s["patient_id"] for s in val_s}
        test_ids = {s["patient_id"] for s in test_s}
        self.assertFalse(train_ids & val_ids)
        self.assertFalse(train_ids & test_ids)
        self.assertFalse(val_ids & test_ids)

    def test_integration_with_agent_and_bridge(self) -> None:
        state = load_scenario("scenarios/scenario_complet.json")
        agent = Agent(state, dl=dl_predict)
        agent.cycle()
        dl_logs = [line for line in agent.journal if line.startswith("[DL] DL_Risk(P1,Critical,")]
        self.assertTrue(dl_logs)

        kb = KnowledgeBase()
        kb.assert_fact(("Patient", "P1"))
        pred = dl_predict(state.patients["P1"].history)
        fact = to_dl_risk_fact("P1", pred)
        kb.assert_fact(fact, source="dl")
        derived, _ = apply_ml_dl_bridge(kb)
        self.assertIn(("CriticalRisk", "P1"), derived)


if __name__ == "__main__":
    unittest.main()
