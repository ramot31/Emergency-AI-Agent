# Chapitre Bloc 2 — Raisonnement logique, déduction, unification, résolution et pont ML/DL (Membre 2)

> **Avertissement** : ce système est un prototype pédagogique de simulation d'aide à la décision aux urgences hospitalières. Il n'a **aucun usage médical réel**.

---

## 1. Architecture du module logique

Le Bloc 2 est structuré en six modules Python autonomes (sans bibliothèque de logique externe telle que SymPy ou pyDatalog) et s'intègre directement dans l'orchestrateur `Agent` du Membre 1 ([agent.py](file:///c:/Users/USER/Desktop/tp%20AI/projet/agent.py)) :

| Fichier | Rôle principal |
|---|---|
| [lexique_predicats.md](file:///c:/Users/USER/Desktop/tp%20AI/projet/lexique_predicats.md) | Référentiel partagé des prédicats, arités, sources et format exact `ML_Risk/3` / `DL_Risk/3` |
| [unification.py](file:///c:/Users/USER/Desktop/tp%20AI/projet/unification.py) | Algorithme d'unification de Robinson avec *occurs check*, composition de substitutions et renommage |
| [rules.py](file:///c:/Users/USER/Desktop/tp%20AI/projet/rules.py) | Base de 28 règles logiques (clauses de Horn avec variables) |
| [kb.py](file:///c:/Users/USER/Desktop/tp%20AI/projet/kb.py) | Base de connaissances (`KnowledgeBase`), maintien de cohérence (TMS), `update(event)` et `reason(facts)` |
| [inference.py](file:///c:/Users/USER/Desktop/tp%20AI/projet/inference.py) | Chaînage avant jusqu'au point fixe (avec justifications) et chaînage arrière optionnel |
| [resolution.py](file:///c:/Users/USER/Desktop/tp%20AI/projet/resolution.py) | Conversion FNC complète (FNN, standardisation, skolémisation) et résolution par réfutation (*Set of Support*) |
| [ml_dl_bridge.py](file:///c:/Users/USER/Desktop/tp%20AI/projet/ml_dl_bridge.py) | Pont ML/DL → KB avec seuils justifiés et résolution tracée des divergences |
| [requirements_export.py](file:///c:/Users/USER/Desktop/tp%20AI/projet/requirements_export.py) | Export des exigences vers le Membre 5 (`export_requirements`) et adaptateur recherche |

---

## 2. Base de connaissances et réutilisation des faits du Membre 1

Les faits produits par [facts.py](file:///c:/Users/USER/Desktop/tp%20AI/projet/facts.py) (`state_to_facts`) sont des tuples Python `(Prédicat, arg1, ...)`. Le module Membre 2 réutilise directement sans redéfinition les seuils et prédicats discrétisés de [environment.py](file:///c:/Users/USER/Desktop/tp%20AI/projet/environment.py) :
- `LowBloodPressure(p)` (`systolic_bp < 90`), `HighBloodPressure(p)` (`systolic_bp > 140`)
- `LowOxygen(p)` (`spo2 < 92`), `Tachycardia(p)` (`heart_rate > 100`), `Fever(p)` (`temperature >= 38.0`)
- `HasSymptom(p, s)`, `PatientWaiting(p)`, `UnderTreatment(p)`, `Available(r)`

La KB sépare les **faits de base** (`base_facts`, étiquetés par source : `environment`, `ml`, `dl`) et les **faits dérivés** (`derived_facts`), chaque fait dérivé conservant une liste de `Justification(rule_id, premises, substitution_str)`. Aucune conclusion (`HighRisk`, `CriticalRisk`, `RequiresICU`, etc.) n'est insérée manuellement : toutes sont dérivées.

---

## 3. Base de règles logiques (`rules.py`)

28 règles (`R01` à `R28`) couvrent :
- **Discrétisation complémentaire** (paramètres non discrétisés par M1) : `R01` (`Age >= 65 -> Elderly`), `R02` (`RespiratoryRate > 22 -> Tachypnea`), `R03` (`OxygenSaturation < 90 -> SevereHypoxia`).
- **Risque élevé (`HighRisk`)** : `R04` (`ChestPain ∧ LowBloodPressure`), `R05` (`LowOxygen ∧ Tachycardia`), `R06` (`Elderly ∧ Fever ∧ LowOxygen`), `R07` (`ShortnessOfBreath ∧ Tachypnea`), `R08` (`ChestPain ∧ HighBloodPressure`).
- **Risque critique (`CriticalRisk`)** : `R09` (`LowBloodPressure ∧ SevereHypoxia`), `R10` (`ChestPain ∧ LowOxygen ∧ Elderly`), `R11` (`Confusion ∧ LowBloodPressure`), `R12` (`CriticalRisk -> HighRisk`).
- **Risque modéré (`ModerateRisk`)** : `R13` (`Patient ∧ Fever`).
- **Prise en charge & ressources** : `R14` (`HighRisk -> RequiresUrgentEvaluation`), `R15` (`CriticalRisk -> RequiresICU`), `R16` (`CriticalRisk -> MonitoringRequired`), `R22`/`R23` (`RequiresEmergencyBed`), `R24` (`PatientWaiting -> RequiresExamRoom`).
- **Spécialistes requis** : `R17` (`Cardiology`), `R18` (`Pulmonology`), `R19`/`R20` (`General`), `R21` (`Neurology`).
- **Priorités** : `R25`–`R28` (`Priority(x, 1..4)`).

---

## 4. Algorithmes implémentés

### 4.1 Unification (`unification.py`)
Implémentation de l'algorithme de Robinson gérant `Var`, constantes typées (`bool` distingué de `int`), `Compound` et atomes tuples `(Pred, ...)`. La vérification d'occurrence (`occurs_in`) empêche les unifications cycliques telles que `x = f(x)`. La composition `compose_subst` et la normalisation `normalize_subst` résolvent les chaînes de variables (`x -> y -> P1`).

### 4.2 Déduction par chaînage avant et arrière (`inference.py`)
- **Chaînage avant (`forward_chain`)** : itère jusqu'au point fixe sur les règles en unifiant leurs prémisses contre l'instantané courant et en vérifiant les gardes arithmétiques. Chaque dérivation enregistre sa justification, permettant `kb.explain(fact)` et l'export Markdown.
- **Chaînage arrière (`backward_chain`)** : prouve un but de haut en bas avec renommage des variables par règle, détection de cycles et profondeur bornée.

### 4.3 Mise en clauses FNC et résolution par réfutation (`resolution.py`)
1. Conversion de formules FOL générales (`Atom`, `Not`, `And`, `Or`, `Implies`, `Iff`, `Forall`, `Exists`) en FNC en 5 passes : élimination de `↔` et `→`, Forme Normale Négative, standardisation des variables, skolémisation (`sk_i` ou `sk_i(vars)`), distribution de `∨` sur `∧`.
2. Négation de la requête placée dans le **Set of Support (SoS)**.
3. Résolution binaire avec unification, élimination des tautologies, détection des doublons modulo renommage de variables (`_canonical_signature`), file de priorité favorisant les clauses courtes (*unit preference*) et bornes `max_steps` / `max_depth`.
4. Extraction du chemin minimal de preuve jusqu'à `□` (`_extract_proof_path`).

---

## 5. Pont ML/DL et gestion des conflits (`ml_dl_bridge.py`)

### Seuils paramétrables (`BridgeThresholds`)
- `dl_critical_threshold = 0.80` et `ml_critical_threshold = 0.80` : un niveau `Critical` suffisamment confiant entraîne `CriticalRisk(p)`.
- `dl_high_threshold = 0.75` et `ml_high_threshold = 0.75` → `HighRisk(p)`.
- `medium_threshold = 0.60` → `ModerateRisk(p)`.
- `ml_chest_pain_threshold = 0.70` : `ML_Risk(p, High|Critical, >= 0.70)` combiné à `HasSymptom(p, ChestPain)` dérive `RequiresUrgentEvaluation(p)`.

### Résolution de conflit ML vs DL
Lorsque `ML_Risk` et `DL_Risk` divergent pour un même patient, un fait `ResourceConflictMLDL(p)` est tracé et la règle suivante s'applique :
1. Si la confiance DL atteint `dl_priority_threshold` (`0.80`), le **DL prime** (car il exploite l'évolution temporelle), sauf garde-fou de précaution si ML prédit un niveau strictement supérieur avec probabilité `>= 0.85`.
2. Si la confiance DL est inférieure à `0.80`, on applique le **principe de précaution** en retenant le niveau de gravité le plus élevé entre ML et DL.

---

## 6. Mise à jour dynamique et maintien de cohérence (TMS)

La méthode `kb.update(event)` supporte les événements du Membre 1 (`resource_unavailable`, `resource_available`, `new_patient`, `new_measurement`), l'alias du contrat PDF (`icu_unavailable`), ainsi que `ml_prediction` et `dl_prediction`. Lors d'un retrait (`retract_fact`), `_propagate_retraction` vérifie les justifications restantes de chaque fait dérivé et invalide en cascade toute conclusion qui n'a plus aucun support valide, tout en préservant celles disposant d'une justification alternative.

---

## 7. Export vers le Membre 5 et intégration avec l'agent

`export_requirements(kb)` retourne pour chaque patient en attente un dictionnaire conforme au contrat n°5 (`priorité` 1..4, `usi` `"oui"`/`"non"`, `spécialité` principale selon préséance clinique, `spécialités`, `type_lit`, `salle_examen`, `évaluation_urgente`). `export_search_requirements(kb)` fournit l'adaptateur pour `agent.py` / `stubs.search_allocation`, et `reason(facts)` remplace `stubs.reason` avec exactement la même signature.

---

## 8. Exemples de traces

Les 5 traces commentées générées automatiquement par le moteur sont disponibles dans [docs/traces/](file:///c:/Users/USER/Desktop/tp%20AI/projet/docs/traces) :
1. [01_chainage_avant_p1_p2_p3.md](file:///c:/Users/USER/Desktop/tp%20AI/projet/docs/traces/01_chainage_avant_p1_p2_p3.md)
2. [02_resolution_requires_icu_p1.md](file:///c:/Users/USER/Desktop/tp%20AI/projet/docs/traces/02_resolution_requires_icu_p1.md)
3. [03_resolution_requires_specialist_p1.md](file:///c:/Users/USER/Desktop/tp%20AI/projet/docs/traces/03_resolution_requires_specialist_p1.md)
4. [04_resolution_non_derivable_p2_p3.md](file:///c:/Users/USER/Desktop/tp%20AI/projet/docs/traces/04_resolution_non_derivable_p2_p3.md)
5. [05_conflit_mldl_et_mise_a_jour_tms.md](file:///c:/Users/USER/Desktop/tp%20AI/projet/docs/traces/05_conflit_mldl_et_mise_a_jour_tms.md)

---

## 9. Limites du raisonnement symbolique dans ce contexte

1. **Rigidité des seuils et effet de frontière** : un patient avec SpO2 = 92 % vs 91 % bascule de `Normal` à `LowOxygen` de manière binaire (d'où la complémentarité indispensable avec le ML/DL).
2. **Incomplétude des règles expertes** : les règles de Horn ne peuvent pas énumérer toutes les combinaisons cliniques atypiques ni les tendances temporelles subtiles.
3. **Absence de négation par l'échec générale en FNC pure** : les règles positives peuvent dériver plusieurs niveaux de priorité (`Priority(P1,1)` et `Priority(P1,4)`), résolus ensuite par sélection du minimum dans `export_requirements`.
4. **Complexité de la résolution du premier ordre** : sur une base volumineuse, la résolution est semi-décidable et requiert des stratégies de contrôle (*Set of Support*, préférence unitaire, bornes d'étapes/profondeur).
5. **Périmètre pédagogique** : les règles sont simplifiées et destinées uniquement à la démonstration de coopération multi-modules d'IA.
