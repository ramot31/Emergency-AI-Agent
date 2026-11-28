"""Module Machine Learning du membre 3.

Ce paquet prédit le risque à partir d'un instantané clinique. Il ne traite pas
l'historique temporel : celui-ci relève du module Deep Learning (membre 4).
"""

from .predict_ml import format_ml_risk, predict_risk, to_ml_risk_fact

__all__ = ["predict_risk", "to_ml_risk_fact", "format_ml_risk"]
