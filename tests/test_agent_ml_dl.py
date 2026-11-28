"""Tests unitaires : branchement des modules ML (Membre 3) et DL (Membre 4) dans l'orchestrateur Agent."""
import json

import agent as agent_module
import stubs
from agent import Agent
from interface import load_scenario

SCENARIO = "scenarios/scenario_complet.json"


def new_agent(**kw):
    return Agent(load_scenario(SCENARIO), **kw)


def test_agent_utilise_les_modules_reels_par_defaut():
    from ml.predict_ml import ml_predict
    from dl.predict_dl import dl_predict
    a = new_agent()
    assert a.ml is ml_predict
    assert a.dl is dl_predict
    assert "ML = réel | DL = réel" in a.journal[0]


def test_injection_de_fonctions_personnalisees():
    ml = lambda patient: ("Low", 0.5)
    dl = lambda history: None
    a = new_agent(ml=ml, dl=dl)
    assert a.ml is ml and a.dl is dl
    assert "injecté" in a.journal[0]


def test_repli_sur_stub_si_module_indisponible(monkeypatch):
    def boom():
        raise ImportError("module absent (simulation)")
    monkeypatch.setattr(agent_module, "_import_ml", boom)
    monkeypatch.setattr(agent_module, "_import_dl", boom)
    a = new_agent()
    assert a.ml is stubs.ml_predict and a.dl is stubs.dl_predict
    assert "ML = stub | DL = stub" in a.journal[0]


def test_cycle_insere_les_predictions_ml_et_dl():
    a = new_agent(); a.cycle()
    log = "\n".join(a.journal)
    assert "ML_Risk(P1,High," in log
    assert "DL_Risk(P1,Critical," in log
    assert "ML_Risk(P2,Low," in log and "ML_Risk(P3,Low," in log
    assert "DL_Risk(P2" not in log          # moins de 3 mesures : le DL renvoie None


def test_p1_obtient_le_lit_usi_avec_les_vrais_modeles():
    a = new_agent(); a.cycle()
    assert ("ReserveICUBed", "P1", "ICU1") in a.plan
    assert ("AssignDoctor", "P1", "D1") in a.plan        # cardiologue


def test_replanification_quand_usi_indisponible():
    a = new_agent(); a.cycle()
    a.execute_next(); a.execute_next()
    a.handle_event({"type": "resource_unavailable", "id": "ICU1"})
    assert a.plan_is_obsolete()
    a.run_plan()
    assert a.state.assignments["P1"]["bed"] != "ICU1"
    assert any("Plan obsolète" in l for l in a.journal)


def test_arrivee_patient_critique_p4():
    a = new_agent()
    a.handle_event(json.load(open("scenarios/event_nouveau_patient_critique.json", encoding="utf-8")))
    a.cycle()
    log = "\n".join(a.journal)
    assert "ML_Risk(P4,Critical," in log and "DL_Risk(P4,Critical," in log
    assert ("ReserveICUBed", "P1", "ICU1") in a.plan      # un seul lit USI : P1, arrivé avant, le garde
    assert not any(act[0] == "ReserveICUBed" and act[1] == "P4" for act in a.plan)
