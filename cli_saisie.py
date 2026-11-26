"""Formulaire en ligne de commande : saisit un patient et l'ajoute à l'état (événement new_patient).
Usage : python cli_saisie.py   -> écrit scenarios/event_saisie.json"""
import json

def ask(label, cast=str, default=None):
    v = input(f"{label}" + (f" [{default}]" if default is not None else "") + " : ").strip()
    return cast(v) if v else default

p = {"id": ask("Identifiant (ex. P5)"), "age": ask("Age", int), "heart_rate": ask("Frequence cardiaque", int),
     "systolic_bp": ask("Tension systolique", int), "spo2": ask("SpO2 (%)", int),
     "temperature": ask("Temperature", float, 37.0), "resp_rate": ask("Frequence respiratoire", int, 16),
     "symptoms": [s.strip() for s in (ask("Symptomes (separes par des virgules)", str, "") or "").split(",") if s.strip()]}
json.dump({"type": "new_patient", "patient": p}, open("scenarios/event_saisie.json", "w", encoding="utf-8"), indent=2)
print("Ecrit dans scenarios/event_saisie.json ->  python main_demo.py scenarios/scenario_complet.json scenarios/event_saisie.json")
