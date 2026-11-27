# Trace 5 — Conflit ML/DL et maintien de cohérence (TMS)

> Simulation pédagogique d'aide à la décision aux urgences, sans usage médical réel.

## 1. Conflit initial ML vs DL sur P1
- Fait ML : `ML_Risk(P1,High,0.87)`
- Fait DL : `DL_Risk(P1,Critical,0.91)`
- Décision du pont : `DL_Risk` prime (`0.91 >= 0.80`) -> `CriticalRisk(P1)` puis `RequiresICU(P1)`.

## 2. Événement 1 : `{"type": "icu_unavailable", "id": "ICU1"}`
- Faits retirés / invalidés : `[('Available', 'ICU1')]`
- `Available(ICU1)` présent ? `False`

## 3. Événement 2 : `{"type": "dl_prediction", "patient": "P1", "level": "Medium", "prob": 0.88}`
- Faits retirés / invalidés par le TMS : `[('DL_Risk', 'P1', 'Critical', 0.91), ('ResourceConflictMLDL', 'P1')]`
- Remarque : `CriticalRisk(P1)` reste encore justifié par les constantes (`R09`/`R10`).

## 4. Événement 3 : `{"type": "new_measurement", "patient": "P1", "hr": 82, "bp": 120, "spo2": 98}`
- Faits retirés / invalidés par le TMS : `[('CriticalRisk', 'P1'), ('HeartRate', 'P1', 132), ('LowBloodPressure', 'P1'), ('LowOxygen', 'P1'), ('MonitoringRequired', 'P1'), ('OxygenSaturation', 'P1', 87), ('Priority', 'P1', 1), ('RequiresICU', 'P1'), ('RequiresSpecialist', 'P1', 'Pulmonology'), ('SevereHypoxia', 'P1'), ('Tachycardia', 'P1')]`
- `CriticalRisk(P1)` présent ? `False`
- `RequiresICU(P1)` présent ? `False`

## Commentaire pédagogique
Le système de maintien de cohérence (TMS) préserve les conclusions tant qu'une justification alternative reste valide (étape 3), puis invalide en cascade `CriticalRisk(P1)` et `RequiresICU(P1)` dès que les mesures physiologiques ne soutiennent plus l'état critique (étape 4).
