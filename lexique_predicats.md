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

## Prédicats dérivés, ML et DL (Membre 2)

> **Avertissement** : système de simulation pédagogique d'aide à la décision, sans usage médical réel.

### Format exact des prédictions ML et DL (contrat Membres 3 & 4 -> Membre 2)
- Représentation Python dans la liste `facts` : tuples `("ML_Risk", patient_id, niveau, prob)` et `("DL_Risk", patient_id, niveau, prob)`.
- `patient_id` : chaîne (`str`, ex. `"P1"`).
- `niveau` : chaîne parmi `"Low"`, `"Medium"`, `"High"`, `"Critical"` (casse exacte en `PascalCase`).
- `prob` : flottant (`float`) dans `[0.0, 1.0]` (ex. `0.87`, `0.91`).
- En cas d'absence de séquence temporelle suffisante (ex. `< 3` mesures), `dl_predict` retourne `None` et aucun fait `DL_Risk` n'est ajouté.

### Table complète des prédicats ML, DL, seuils et dérivés

| Prédicat | Arité | Signification | Source | Format Python (tuple) |
|---|---|---|---|---|
| `ML_Risk(p,niveau,prob)` | 3 | Prédiction du niveau de risque par Machine Learning sur l'état courant | ML (Membre 3) | `("ML_Risk", "P1", "High", 0.87)` |
| `DL_Risk(p,niveau,prob)` | 3 | Prédiction du niveau de risque par Deep Learning sur la séquence temporelle | DL (Membre 4) | `("DL_Risk", "P1", "Critical", 0.91)` |
| `Elderly(p)` | 1 | Patient âgé (`Age >= 65`), dérivé depuis `Age(p,n)` | dérivé (Membre 2 depuis environnement) | `("Elderly", "P1")` |
| `Tachypnea(p)` | 1 | Fréquence respiratoire élevée (`RespiratoryRate > 22`), dérivé | dérivé (Membre 2 depuis environnement) | `("Tachypnea", "P1")` |
| `SevereHypoxia(p)` | 1 | Saturation critique (`OxygenSaturation < 90`), dérivé | dérivé (Membre 2 depuis environnement) | `("SevereHypoxia", "P1")` |
| `HighRisk(p)` | 1 | Patient à risque élevé nécessitant prise en charge prioritaire | dérivé (Membre 2) | `("HighRisk", "P1")` |
| `CriticalRisk(p)` | 1 | Patient en situation critique (risque vital simulé) | dérivé (Membre 2) | `("CriticalRisk", "P1")` |
| `ModerateRisk(p)` | 1 | Patient à risque modéré (fièvre isolée ou prédiction Medium) | dérivé (Membre 2) | `("ModerateRisk", "P2")` |
| `LowRisk(p)` | 1 | Patient stable sans signe de gravité ni alerte ML/DL | dérivé (Membre 2) | `("LowRisk", "P3")` |
| `RequiresUrgentEvaluation(p)` | 1 | Évaluation médicale urgente requise | dérivé (Membre 2) | `("RequiresUrgentEvaluation", "P1")` |
| `RequiresICU(p)` | 1 | Lit de soins intensifs (USI) requis | dérivé (Membre 2) | `("RequiresICU", "P1")` |
| `RequiresEmergencyBed(p)` | 1 | Lit d'urgence standard (`Emergency`) requis | dérivé (Membre 2) | `("RequiresEmergencyBed", "P2")` |
| `RequiresExamRoom(p)` | 1 | Salle d'examen simple requise | dérivé (Membre 2) | `("RequiresExamRoom", "P3")` |
| `RequiresSpecialist(p,s)` | 2 | Spécialité médicale requise (`Cardiology`, `Pulmonology`, `General`, `InfectiousDisease`, `Gastroenterology`, `Neurology`) | dérivé (Membre 2) | `("RequiresSpecialist", "P1", "Cardiology")` |
| `Priority(p,niveau)` | 2 | Niveau de priorité dérivé : `1` (Critique/Imméd.), `2` (Élevé/Urgent), `3` (Modéré), `4` (Faible/Standard) | dérivé (Membre 2) | `("Priority", "P1", 1)` |
| `MonitoringRequired(p)` | 1 | Surveillance continue recommandée | dérivé (Membre 2) | `("MonitoringRequired", "P1")` |
| `ResourceConflictMLDL(p)` | 1 | Divergence détectée entre `ML_Risk` et `DL_Risk` résolue par le pont ML/DL | dérivé (Membre 2) | `("ResourceConflictMLDL", "P1")` |

### Rappel des prédicats environnement utilisés en prémisses (Membre 1)
- Discretisés par Membre 1 : `LowBloodPressure/1`, `HighBloodPressure/1`, `LowOxygen/1`, `Tachycardia/1`, `Fever/1` (source : environnement).
- Symptômes & statut : `HasSymptom/2`, `PatientWaiting/1`, `UnderTreatment/1`, `Available/1`, `Type/2`, `Specialty/2` (source : environnement).

