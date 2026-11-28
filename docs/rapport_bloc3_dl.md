# Bloc 3 - Deep Learning (Membre 4)

> **Avertissement** : système de simulation pédagogique d'aide à la décision aux urgences, sans aucun usage médical réel.

## 1. Objectif

Le module Deep Learning analyse l'évolution temporelle des constantes vitales d'un patient (`hr`, `bp`, `spo2`) sur plusieurs pas de temps afin de détecter une détérioration qu'un modèle statique (ne voyant que l'instant courant) ne peut pas percevoir. Il produit une évidence probabiliste `DL_Risk(PatientId, Niveau, Probabilité)` consommée par l'orchestrateur (`agent.py`) et le pont logique (`ml_dl_bridge.py`).

## 2. Données séquentielles et prétraitement

* **Jeu de données synthétique** (`dl/data/synthetic_sequences.csv`, généré de manière déterministe avec `RANDOM_STATE = 42`) : **1 600 patients** et **8 737 mesures**, avec des séquences de longueur variable entre 3 et 8 pas de temps.
* **Quatre profils de trajectoires** : `stable` (446), `deterioration_lente` (388), `deterioration_rapide` (385) et `amelioration` (381).
* **Quatre classes communes avec le ML** : `Low` (797), `Critical` (424), `High` (214), `Medium` (165).
* **Variables et features temporelles** (`dl/preprocessing.py`) :
  * Variables brutes issues strictement de `Patient.snapshot()` (`hr`, `bp`, `spo2`).
  * Variations cumulées depuis l'arrivée (`delta_*_t0`) et variations pas à pas (`delta_*_step`), soit 9 features par pas de temps.
* **Prévention des fuites de données** :
  * Découpage stratifié **par `patient_id`** (`train = 1 119`, `val = 241`, `test = 240`), sans aucun patient partagé entre les ensembles.
  * Normalisation z-score ajustée uniquement sur le `train` (`dl/artifacts/scaler_stats.json`) et padding à droite jusqu'à `MAX_SEQ_LEN = 8`.

## 3. Architecture, entraînement et modèle de référence

* **Modèles récurrents comparés** (`dl/model.py`, PyTorch) : **LSTM** et **GRU** à 2 couches (`hidden_size = 48`, `dropout = 0.25`) avec `pack_padded_sequence` pour ignorer le padding.
* **Entraînement** (`dl/train_dl.py`) :
  * Pondération des classes dans `CrossEntropyLoss` (avec priorité accrue sur `Critical` pour limiter les faux négatifs).
  * Optimiseur `AdamW` (`lr = 1.5e-3`, `weight_decay = 1e-4`), *early stopping* sur la perte de validation (`val_loss`, patience = 12), graine fixée à `42`.
  * Courbes sauvegardées dans `dl/artifacts/learning_curves.svg`.
* **Modèle de référence (baseline)** : `RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)` entraîné uniquement sur le **dernier pas de temps** `(hr, bp, spo2)`.

## 4. Résultats et comparaison (jeu de test, 240 patients)

| Modèle | Accuracy | F1 macro | Recall `Critical` | Précision `Critical` | Accuracy (détérioration) |
|---|---|---|---|---|---|
| **LSTM (retenu)** | **88,33 %** | **83,12 %** | **87,50 %** | **96,55 %** | **83,33 %** |
| GRU | 86,25 % | 81,72 % | 85,94 % | 98,21 % | 84,21 % |
| Baseline Random Forest (dernier pas) | 84,17 % | 74,87 % | 89,06 % | 91,94 % | 81,58 % |

* Le **LSTM** est retenu devant le **GRU** (meilleurs accuracy, F1 macro et recall `Critical` sur la validation et le test).
* Face à la **baseline statique**, le LSTM gagne **+4,16 points d'accuracy** et **+8,25 points de F1 macro**, en distinguant beaucoup mieux les trajectoires en cours de dégradation (`Medium` / `High`) avant qu'elles n'atteignent des seuils extrêmes.

## 5. Inférence et re-inférence dynamique

* `predict_deterioration(history)` / `dl_predict(history)` (`dl/predict_dl.py`) : retourne `None` si `len(history) < 3`, sinon `(niveau, probabilité)`.
* Pour le scénario `P1` (`scenarios/scenario_complet.json`), le modèle prédit `Critical` (`0.9957`), produisant `DL_Risk(P1,Critical,1.00)`.
* `update_and_repredict(history, new_measurement)` permet au Membre 6 de recalculer la prédiction dès qu'une nouvelle mesure arrive dans le scénario dynamique.

## 6. Section commune ML vs DL (avec le Membre 3)

| Critère | Machine Learning (Membre 3) | Deep Learning (Membre 4) |
|---|---|---|
| **Données d'entrée** | Instantané courant (6 constantes + 8 symptômes) | Série temporelle (`hr`, `bp`, `spo2` sur 3 à 8 pas) |
| **Modèle retenu** | Decision Tree (parmi 5 modèles tabulaires) | LSTM PyTorch (comparé à GRU et RF dernier pas) |
| **Force principale** | Rapide, interprétable, fonctionne dès l'arrivée ($t_0$) | Détecte la dynamique de détérioration ou d'amélioration |
| **Cas où il n'agit pas** | — | Séquence $< 3$ mesures (`dl_predict` retourne `None`) |
| **Intégration KB** | `ML_Risk(p, niveau, prob)` | `DL_Risk(p, niveau, prob)` (prioritaire si `prob >= 0.80`) |

## 7. Limites

* Les données sont entièrement synthétiques et restreintes à 3 constantes (`hr`, `bp`, `spo2`) afin de respecter le contrat de `Patient.snapshot()`.
* Les probabilités issues du `softmax` ne sont pas calibrées pour un usage clinique réel.
* La taille modérée du jeu de données (1 600 patients) impose une régularisation stricte (`dropout`, `weight_decay`, `early stopping`) et interdit toute extrapolation médicale réelle.
