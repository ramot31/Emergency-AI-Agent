# Lexique des prédicats (référence pour toute l'équipe)

## Faits produits par l'environnement (Membre 1)
| Prédicat | Arité | Signification |
|---|---|---|
| Patient(p) | 1 | p est un patient |
| Age(p,n) / HeartRate(p,n) / OxygenSaturation(p,n) / RespiratoryRate(p,n) | 2 | mesures entières |
| Temperature(p,x) | 2 | température en °C |
| LowBloodPressure(p) / HighBloodPressure(p) | 1 | tension discrétisée (seuils dans environment.py) |
| LowOxygen(p) / Tachycardia(p) / Fever(p) | 1 | indicateurs discrétisés |
| HasSymptom(p,s) | 2 | symptôme (ChestPain, ShortnessOfBreath...) |
| PatientWaiting(p) / UnderTreatment(p) | 1 | statut |
| Doctor(d) / Nurse(n) / Bed(b) / Room(r) | 1 | type de ressource |
| Specialty(d,s) | 2 | spécialité du médecin |
| Type(b,t) | 2 | type de lit : Emergency ou ICU |
| Available(r) | 1 | ressource disponible (absent = indisponible) |
| DoctorAssigned(p,d) / BedAssigned(p,b) / RoomAssigned(p,r) | 2 | affectations |

## Faits produits par l'apprentissage (Membres 3 et 4)
| Prédicat | Arité | Signification |
|---|---|---|
| ML_Risk(p,niveau,prob) | 3 | niveau parmi Low, Medium, High, Critical ; prob dans [0,1] |
| DL_Risk(p,niveau,prob) | 3 | idem, issu de l'analyse temporelle |

## Faits dérivés par le moteur logique (Membre 2)
HighRisk(p), CriticalRisk(p), RequiresUrgentEvaluation(p), RequiresICU(p), RequiresSpecialist(p,s)
