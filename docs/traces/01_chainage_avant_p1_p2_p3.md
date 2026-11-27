# Trace 1 — Chaînage avant complet (scénario P1 / P2 / P3)

> Simulation pédagogique d'aide à la décision aux urgences, sans usage médical réel.

- Faits de base : **53**
- Faits dérivés : **28**

## Étapes d'inférence

1. `[BRIDGE_CONFLICT] Conflit ML=High(0.87) vs DL=Critical(0.91) : DL prime (0.91 >= 0.80, analyse temporelle)`
2. `[BRIDGE_ML_DL] ML_Risk(P1,High,0.87) ^ DL_Risk(P1,Critical,0.91) -> CriticalRisk(P1) {p/P1} (DL)`
3. `[BRIDGE_ML_CHESTPAIN] ML_Risk(P1,High,0.87) ^ HasSymptom(P1,ChestPain) -> RequiresUrgentEvaluation(P1) {p/P1} (ML=High(0.87) + ChestPain)`
4. `--- Itération 1 (17 nouveaux faits) ---`
5. `[R01] Age(P1,72) -> Elderly(P1) {a/72, x/P1}`
6. `[R02] RespiratoryRate(P1,26) -> Tachypnea(P1) {r/26, x/P1}`
7. `[R03] OxygenSaturation(P1,87) -> SevereHypoxia(P1) {s/87, x/P1}`
8. `[R04] HasSymptom(P1,ChestPain) ^ LowBloodPressure(P1) -> HighRisk(P1) {x/P1}`
9. `[R13] Patient(P1) ^ Fever(P1) -> ModerateRisk(P1) {x/P1}`
10. `[R13] Patient(P2) ^ Fever(P2) -> ModerateRisk(P2) {x/P2}`
11. `[R15] CriticalRisk(P1) -> RequiresICU(P1) {x/P1}`
12. `[R16] CriticalRisk(P1) -> MonitoringRequired(P1) {x/P1}`
13. `[R18] HasSymptom(P1,ShortnessOfBreath) ^ LowOxygen(P1) -> RequiresSpecialist(P1,Pulmonology) {x/P1}`
14. `[R19] Patient(P3) ^ HasSymptom(P3,AbdominalPain) -> RequiresSpecialist(P3,General) {x/P3}`
15. `[R24] PatientWaiting(P1) -> RequiresExamRoom(P1) {x/P1}`
16. `[R24] PatientWaiting(P2) -> RequiresExamRoom(P2) {x/P2}`
17. `[R24] PatientWaiting(P3) -> RequiresExamRoom(P3) {x/P3}`
18. `[R25] CriticalRisk(P1) -> Priority(P1,1) {x/P1}`
19. `[R28] Patient(P1) -> Priority(P1,4) {x/P1}`
20. `[R28] Patient(P2) -> Priority(P2,4) {x/P2}`
21. `[R28] Patient(P3) -> Priority(P3,4) {x/P3}`
22. `--- Itération 2 (8 nouveaux faits) ---`
23. `[R17] HighRisk(P1) ^ HasSymptom(P1,ChestPain) -> RequiresSpecialist(P1,Cardiology) {x/P1}`
24. `[R20] ModerateRisk(P1) ^ Fever(P1) -> RequiresSpecialist(P1,General) {x/P1}`
25. `[R20] ModerateRisk(P2) ^ Fever(P2) -> RequiresSpecialist(P2,General) {x/P2}`
26. `[R22] HighRisk(P1) -> RequiresEmergencyBed(P1) {x/P1}`
27. `[R23] ModerateRisk(P2) -> RequiresEmergencyBed(P2) {x/P2}`
28. `[R26] HighRisk(P1) -> Priority(P1,2) {x/P1}`
29. `[R27] ModerateRisk(P1) -> Priority(P1,3) {x/P1}`
30. `[R27] ModerateRisk(P2) -> Priority(P2,3) {x/P2}`

## Commentaire pédagogique

- **P1** (72 ans, douleur thoracique, hypotension, SpO2 87 %, `ML_Risk(P1,High,0.87)`, `DL_Risk(P1,Critical,0.91)`) déclenche le pont ML/DL (le DL prime avec `0.91 >= 0.80`), dérive `CriticalRisk(P1)`, `HighRisk(P1)`, `RequiresUrgentEvaluation(P1)`, `RequiresICU(P1)`, `RequiresSpecialist(P1,Cardiology)` et `Priority(P1,1)`.
- **P2** (45 ans, fièvre 38.4 °C, constantes stables) dérive `ModerateRisk(P2)`, `RequiresEmergencyBed(P2)` et `Priority(P2,3)`, **sans** déclencher l'USI.
- **P3** (30 ans, douleur abdominale stable) reste à `Priority(P3,4)` avec `RequiresExamRoom(P3)` et `RequiresSpecialist(P3,General)`, **sans** USI.
