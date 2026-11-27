# Trace 2 — Résolution par réfutation : `RequiresICU(P1)?` (succès)

> Simulation pédagogique d'aide à la décision aux urgences, sans usage médical réel.

- **Requête** : `RequiresICU(P1)?`
- **Verdict** : **PROUVÉ (□)** — Clause vide atteinte en 8 étape(s)
- **Étapes explorées** : 8

## Clauses initiales (FNC)

- `C1` : `Age(P1,72)` *(fait)*
- `C2` : `Age(P2,45)` *(fait)*
- `C3` : `Age(P3,30)` *(fait)*
- `C4` : `Available(D1)` *(fait)*
- `C5` : `Available(D2)` *(fait)*
- `C6` : `Available(E1)` *(fait)*
- `C7` : `Available(E2)` *(fait)*
- `C8` : `Available(ICU1)` *(fait)*
- `C9` : `Available(N1)` *(fait)*
- `C10` : `Available(R1)` *(fait)*
- `C11` : `Bed(E1)` *(fait)*
- `C12` : `Bed(E2)` *(fait)*
- `C13` : `Bed(ICU1)` *(fait)*
- `C14` : `DL_Risk(P1,Critical,0.91)` *(fait)*
- `C15` : `Doctor(D1)` *(fait)*
- `C16` : `Doctor(D2)` *(fait)*
- `C17` : `Fever(P1)` *(fait)*
- `C18` : `Fever(P2)` *(fait)*
- `C19` : `HasSymptom(P1,ChestPain)` *(fait)*
- `C20` : `HasSymptom(P1,ShortnessOfBreath)` *(fait)*
- `C21` : `HasSymptom(P2,Fever)` *(fait)*
- `C22` : `HasSymptom(P3,AbdominalPain)` *(fait)*
- `C23` : `HeartRate(P1,132)` *(fait)*
- `C24` : `HeartRate(P2,88)` *(fait)*
- `C25` : `HeartRate(P3,80)` *(fait)*
- `C26` : `LowBloodPressure(P1)` *(fait)*
- `C27` : `LowOxygen(P1)` *(fait)*
- `C28` : `ML_Risk(P1,High,0.87)` *(fait)*
- `C29` : `ML_Risk(P2,Low,0.8)` *(fait)*
- `C30` : `ML_Risk(P3,Low,0.8)` *(fait)*
- `C31` : `Nurse(N1)` *(fait)*
- `C32` : `OxygenSaturation(P1,87)` *(fait)*
- `C33` : `OxygenSaturation(P2,97)` *(fait)*
- `C34` : `OxygenSaturation(P3,98)` *(fait)*
- `C35` : `Patient(P1)` *(fait)*
- `C36` : `Patient(P2)` *(fait)*
- `C37` : `Patient(P3)` *(fait)*
- `C38` : `PatientWaiting(P1)` *(fait)*
- `C39` : `PatientWaiting(P2)` *(fait)*
- `C40` : `PatientWaiting(P3)` *(fait)*
- `C41` : `RespiratoryRate(P1,26)` *(fait)*
- `C42` : `RespiratoryRate(P2,16)` *(fait)*
- `C43` : `RespiratoryRate(P3,16)` *(fait)*
- `C44` : `Room(R1)` *(fait)*
- `C45` : `Specialty(D1,Cardiology)` *(fait)*
- `C46` : `Specialty(D2,General)` *(fait)*
- `C47` : `Tachycardia(P1)` *(fait)*
- `C48` : `Temperature(P1,39.0)` *(fait)*
- `C49` : `Temperature(P2,38.4)` *(fait)*
- `C50` : `Temperature(P3,36.8)` *(fait)*
- `C51` : `Type(E1,Emergency)` *(fait)*
- `C52` : `Type(E2,Emergency)` *(fait)*
- `C53` : `Type(ICU1,ICU)` *(fait)*
- `C54` : `¬HasSymptom(x_1,ChestPain) ∨ ¬LowBloodPressure(x_1) ∨ HighRisk(x_1)` *(R04)*
- `C55` : `¬LowOxygen(x_1) ∨ ¬Tachycardia(x_1) ∨ HighRisk(x_1)` *(R05)*
- `C56` : `¬Elderly(x_1) ∨ ¬Fever(x_1) ∨ ¬LowOxygen(x_1) ∨ HighRisk(x_1)` *(R06)*
- `C57` : `¬HasSymptom(x_1,ShortnessOfBreath) ∨ ¬Tachypnea(x_1) ∨ HighRisk(x_1)` *(R07)*
- `C58` : `¬HasSymptom(x_1,ChestPain) ∨ ¬HighBloodPressure(x_1) ∨ HighRisk(x_1)` *(R08)*
- `C59` : `¬LowBloodPressure(x_1) ∨ ¬SevereHypoxia(x_1) ∨ CriticalRisk(x_1)` *(R09)*
- `C60` : `¬HasSymptom(x_1,ChestPain) ∨ ¬LowOxygen(x_1) ∨ ¬Elderly(x_1) ∨ CriticalRisk(x_1)` *(R10)*
- `C61` : `¬HasSymptom(x_1,Confusion) ∨ ¬LowBloodPressure(x_1) ∨ CriticalRisk(x_1)` *(R11)*
- `C62` : `¬CriticalRisk(x_1) ∨ HighRisk(x_1)` *(R12)*
- `C63` : `¬Patient(x_1) ∨ ¬Fever(x_1) ∨ ModerateRisk(x_1)` *(R13)*
- `C64` : `¬HighRisk(x_1) ∨ RequiresUrgentEvaluation(x_1)` *(R14)*
- `C65` : `¬CriticalRisk(x_1) ∨ RequiresICU(x_1)` *(R15)*
- `C66` : `¬CriticalRisk(x_1) ∨ MonitoringRequired(x_1)` *(R16)*
- `C67` : `¬HighRisk(x_1) ∨ ¬HasSymptom(x_1,ChestPain) ∨ RequiresSpecialist(x_1,Cardiology)` *(R17)*
- `C68` : `¬HasSymptom(x_1,ShortnessOfBreath) ∨ ¬LowOxygen(x_1) ∨ RequiresSpecialist(x_1,Pulmonology)` *(R18)*
- `C69` : `¬Patient(x_1) ∨ ¬HasSymptom(x_1,AbdominalPain) ∨ RequiresSpecialist(x_1,General)` *(R19)*
- `C70` : `¬ModerateRisk(x_1) ∨ ¬Fever(x_1) ∨ RequiresSpecialist(x_1,General)` *(R20)*
- `C71` : `¬Patient(x_1) ∨ ¬HasSymptom(x_1,Confusion) ∨ RequiresSpecialist(x_1,Neurology)` *(R21)*
- `C72` : `¬HighRisk(x_1) ∨ RequiresEmergencyBed(x_1)` *(R22)*
- `C73` : `¬ModerateRisk(x_1) ∨ RequiresEmergencyBed(x_1)` *(R23)*
- `C74` : `¬PatientWaiting(x_1) ∨ RequiresExamRoom(x_1)` *(R24)*
- `C75` : `¬CriticalRisk(x_1) ∨ Priority(x_1,1)` *(R25)*
- `C76` : `¬HighRisk(x_1) ∨ Priority(x_1,2)` *(R26)*
- `C77` : `¬ModerateRisk(x_1) ∨ Priority(x_1,3)` *(R27)*
- `C78` : `¬Patient(x_1) ∨ Priority(x_1,4)` *(R28)*
- `C79` : `¬Age(P1,72) ∨ Elderly(P1)` *(R01(inst))*
- `C80` : `¬RespiratoryRate(P1,26) ∨ Tachypnea(P1)` *(R02(inst))*
- `C81` : `¬OxygenSaturation(P1,87) ∨ SevereHypoxia(P1)` *(R03(inst))*
- `C82` : `¬RequiresICU(P1)` *(négation de la requête — SoS)*

## Chemin de réfutation jusqu'à `□`

1. `Étape 1 [C83] : C82 + C65 sur (¬RequiresICU(P1) , RequiresICU(x_1_b65)) unif={x_1_b65/P1} => ¬CriticalRisk(P1)`
1. `Étape 2 [C84] : C83 + C59 sur (¬CriticalRisk(P1) , CriticalRisk(x_1_b140)) unif={x_1_b140/P1} => ¬LowBloodPressure(P1) ∨ ¬SevereHypoxia(P1)`
1. `Étape 5 [C87] : C84 + C26 sur (¬LowBloodPressure(P1) , LowBloodPressure(P1)) unif={} => ¬SevereHypoxia(P1)`
1. `Étape 7 [C89] : C87 + C81 sur (¬SevereHypoxia(P1) , SevereHypoxia(P1)) unif={} => ¬OxygenSaturation(P1,87)`
1. `Étape 8 [C90] : C89 + C32 sur (¬OxygenSaturation(P1,87) , OxygenSaturation(P1,87)) unif={} => □`

## Commentaire pédagogique

La négation de la requête `¬RequiresICU(P1)` est placée dans le *Set of Support*. Par résolution binaire et unification avec les clauses de règles (`R15`, `R09`/`R10`, `R03`) et les faits de `P1`, le moteur atteint la clause vide `□`.
