# Agent intelligent pour la gestion des urgences (simulation, aucun usage médical réel)
Démo environnement : `python main_demo.py scenarios/scenario_complet.json scenarios/event_icu_indisponible.json`
Démo agent complet : `python demo_agent.py`
Tests : `python -m pytest -q`  (ou `python test_env.py`)
Saisie d'un patient : `python cli_saisie.py`
Fichiers : environment.py (état), facts.py (faits), interface.py (JSON/événements), agent.py (orchestrateur),
stubs.py (modules factices à remplacer par les vrais modules des Membres 2 à 6).
