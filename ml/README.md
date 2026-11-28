# Module Machine Learning - Membre 3

Ce module est une simulation académique : il n'est pas destiné à un usage médical réel.

## Périmètre et contrat d'intégration

Le modèle exploite uniquement les constantes courantes du `Patient` du Bloc 1 : `age`, `heart_rate`, `systolic_bp`, `spo2`, `temperature`, `resp_rate` et `symptoms`. Il n'utilise jamais `history` : l'évolution temporelle relève du module DL (membre 4).

L'entrée est soit une instance de `environment.Patient`, soit un dictionnaire ayant ces mêmes clés. La sortie respecte le contrat consommé par `agent.py` :

```python
from ml.predict_ml import predict_risk, to_ml_risk_fact, format_ml_risk

prediction = predict_risk(patient)       # ("High", 0.87)
fact = to_ml_risk_fact(patient.id, prediction)
# ("ML_Risk", "P1", "High", 0.87)  <- format tuple réellement consommé par la KB
print(format_ml_risk(patient.id, prediction))
# ML_Risk(P1,High,0.87)
```

Pour injecter le vrai modèle sans modifier `agent.py`, il suffit de l'injecter à la construction :

```python
from agent import Agent
from ml.predict_ml import ml_predict

agent = Agent(state, ml=ml_predict)
```

Le membre 2 convertit ensuite ce fait en conclusions symboliques avec ses seuils et sa résolution de conflit ML/DL. Le membre 1 reste responsable de `Patient` et de l'orchestration ; ce module ne les modifie pas.

## Données synthétiques

`data/synthetic_triage.csv` est généré de façon déterministe (`random_state=42`) par `data_generation.py`. Il contient 2 400 patients simulés et les quatre classes `Low`, `Medium`, `High`, `Critical`.

Les constantes sont tirées de profils de sévérité simulés. Une règle de score latent, documentée dans `data_generation.py`, produit l'étiquette ; un faible bruit est ajouté. Une cohorte déterministe de profils `High` graves mais non extrêmes, autour du scénario P1, évite d'assimiler ces cas à `Critical` et assure la représentation de chaque classe. Les symptômes sont multi-étiquetés puis encodés en indicateurs binaires. Environ 1,5 % des constantes sont rendues manquantes et 0,5 % reçoivent une valeur aberrante afin d'exercer le pipeline. Les étiquettes sont calculées avant ces dégradations.

Ce jeu ne représente ni une population réelle, ni un protocole de triage médical. Les performances ne sont donc valables que pour cette démonstration pédagogique ; elles ne peuvent pas être généralisées ni utilisées pour soigner ou orienter des patients.

## Prétraitement et évaluation

Le pipeline sauvegardé réalise, exclusivement dans chaque pli d'entraînement :

- bornage des constantes à des plages plausibles simulées ;
- imputation médiane des constantes manquantes ;
- standardisation des constantes ;
- encodage multi-hot stable des symptômes.

Le découpage est stratifié (75 % apprentissage / 25 % test). Cinq modèles sont comparés avec recherche limitée d'hyperparamètres : régression logistique, arbre de décision, forêt aléatoire, KNN et SVM. Les classes sont pondérées pour la régression, les arbres et le SVM ; KNN ne propose pas de pondération par classe, mais compare les votes uniformes et pondérés par distance. KNN et SVM bénéficient de la standardisation des constantes dans chaque pli. Le SVM est calibré par validation croisée interne pour fournir les probabilités attendues par l'interface, sans garantie de calibration clinique. Les hyperparamètres et le modèle sont sélectionnés uniquement par validation croisée stratifiée à quatre plis sur l'apprentissage : rappel `Critical` prioritaire, puis F1 macro pour départager. Le test est calculé après ce choix. Comme le jeu de test et sa graine ont déjà été examinés pendant le développement antérieur, ces résultats ne constituent toutefois pas une estimation strictement indépendante ; un nouveau jeu externe serait nécessaire.

L'entraînement écrit dans `artifacts/` : le pipeline Joblib, un tableau de comparaison, le rapport JSON complet, une matrice de confusion par modèle et une importance par permutation. Les chiffres présentés dans le rapport doivent toujours être lus depuis les artefacts de l'exécution courante, jamais recopiés à la main.

## Exécution reproductible

Depuis la racine `Emergency-AI-Agent` :

```powershell
python -m ml.train_ml
python -m pytest tests/test_ml_module.py -q
python -m pytest -q
```

La première commande crée le dataset si besoin, entraîne les modèles et sauvegarde `ml/artifacts/risk_pipeline.joblib`. Pour régénérer seulement les données :

```powershell
python -m ml.data_generation
```

Les dépendances sont `numpy`, `pandas`, `scikit-learn` et `joblib`. Les graphiques sont des SVG autonomes, générés sans backend graphique.

Les versions testées dans l'environnement existant figurent dans `ml/requirements-ml.txt` (Python 3.13.1) : `python -m pip install -r ml/requirements-ml.txt`. Leur installation et le chargement de l'artefact n'ont pas encore été vérifiés dans un environnement propre. Les artefacts Joblib dépendent des versions Python/scikit-learn et ne doivent être chargés que s'ils proviennent d'une source de confiance.

### Démonstration, explications et évaluation complémentaire

```powershell
python -m ml.demo_cooperation
python -m ml.explain_ml
python -m ml.evaluate_external
python -m ml.analyze_scores
python -m ml.compare_policies
python -m ml.final_policy_evaluation
```

La démonstration cherche dans les données de développement un patient pour lequel la prédiction réelle modifie les exigences exportées par la KB. Elle compare les faits et traces avec et sans ML ; elle ne teste ni le vrai DL ni un algorithme de recherche ou de planification. Le chemin de l'arbre présente les seuils prétraités et leur équivalent approximatif dans l'unité d'origine, après bornage et imputation ; il ne démontre pas un lien causal médical. L'explication est disponible pour l'arbre actuellement retenu.

### Politique de jugement règles / ML

La trace `[JUGEMENT_REGLES_ML]` compare le risque minimal dérivé des seules règles à la prédiction ML et indique le niveau effectivement présent après pont et chaînage. Une conclusion des règles est une borne minimale : `Medium` par règle et `High` par ML ne sont pas nécessairement contradictoires. Si ML indique un niveau inférieur à ce minimum, la trace signale le désaccord et recommande une réévaluation ; les faits des règles demeurent et le calcul de priorité continue. Si les règles ne concluent rien, ce n'est pas une conclusion `Low`. Les scores sous le seuil principal sont signalés, sans exclure l'application d'une autre règle combinée ; les prédictions de format ou score invalide sont ignorées par le pont. Les scores ne sont pas calibrés cliniquement et les seuils sont des paramètres de simulation.

Le jugement est une **trace explicative**, pas un nouveau fait ni une action automatiquement planifiée : aucune réévaluation n'est déclenchée par cette mention. La politique ML/DL existante reste applicable et peut influer sur le niveau retenu. L'agent doit être construit avec `reasoner=kb.reason` et `ml=ml_predict` pour voir cette trace dans son cycle ; par défaut il utilise encore des stubs.

Le pont applique désormais une modération ciblée lorsque seul le ML est présent : une prédiction `High` au-dessus de son seuil devient `Medium` provisoire si les règles ne concluent à aucun niveau de risque. L'absence de conclusion n'est pas une prédiction `Low` : cette limite est une décision opérationnelle, pas un diagnostic. Si une règle conclut au moins `Medium`, `High` est conservé. Une prédiction ML `Critical` acceptée n'est pas abaissée par cette politique et aucune conclusion des règles n'est diminuée. Les autres règles combinées, dont ML + douleur thoracique, et l'arbitrage ML/DL existant restent applicables ; ils peuvent produire d'autres conclusions. La trace indique la modération et la réévaluation recommandée n'est pas programmée automatiquement.

`python -m ml.compare_policies` compare règles seules, ancienne politique et politique modérée en exécutant le vrai pont et les vraies règles sur les observations complètes du jeu synthétique complémentaire ; l'ancienne politique ne diffère que par la désactivation de la modération. Les éventuelles conclusions d'évaluation urgente sont comptabilisées séparément du niveau de risque. Ce jeu est déjà connu et a servi à examiner le compromis : il ne permet pas de validation indépendante ni de démontrer que la politique est optimale. La comparaison n'appelle ni le DL ni l'allocation et exclut les mesures manquantes.

`python -m ml.analyze_scores` enregistre `score_analysis.json` et `threshold_analysis.csv` : répartition des scores de la classe prédite, taux de bonnes classifications et comparaison descriptive de seuils du ML seul. Une prédiction non acceptée n'est pas une classe `Low` et cette analyse ne reproduit pas les décisions du pont. Le Brier multiclasse est la moyenne de la somme des écarts quadratiques aux étiquettes binaires ; plus faible est préférable sur un même jeu. Ces mesures sur données synthétiques déjà consultées ne valident pas les scores ni un seuil pour la pratique.

`python -m ml.final_policy_evaluation` évalue les politiques sur un tirage distinct créé après le choix de la modération (graine 734921). Le fichier est maintenant consulté et ne doit plus servir de jeu final vierge pour un futur réglage. Il utilise les mêmes règles de génération et ne constitue pas une validation externe clinique ; les mesures manquantes sont exclues de cette comparaison.

Pour examiner les données avant utilisation, `inspect_patient(patient)` renvoie des avertissements pour les champs numériques manquants, invalides ou bornés et les symptômes non reconnus. `predict_risk_with_diagnostics(patient)` associe ces avertissements à la prédiction, mais ne les injecte pas automatiquement dans `Agent` : l'interface standard `predict_risk` reste inchangée et l'intégration de ces avertissements est à coordonner avec les autres membres. Une clé numérique complètement absente est refusée ; `None` comme valeur est imputée par le pipeline. Un symptôme absent est traité comme non observé, ce qui ne prouve pas son absence.

L'évaluation complémentaire utilise 1 200 observations générées avec une autre graine et un modèle déjà entraîné. `independent_evaluation.json` contient les métriques et `independent_critical_errors.csv` détaille les cas critiques manqués. L'absence de lignes identiques ne prouve pas une indépendance clinique : le jeu provient du même générateur et conserve ses hypothèses, notamment les profils autour de P1. Le minimum de 240 profils autour de P1 entraîne une proportion de ces profils supérieure à celle du jeu d'entraînement. Ne pas utiliser ces résultats pour sélectionner ou régler le modèle ; toute modification ultérieure informerait sur ce jeu désormais observé.

## Points d'intégration restants

- Membre 4 : conserver exactement les quatre libellés `Low`, `Medium`, `High`, `Critical`, sans confondre ce modèle instantané avec son analyse d'historique.
- Membre 2 : consommer le tuple `("ML_Risk", patient_id, level, probability)` ; ses seuils du pont ML/DL restent sa responsabilité.
- Membre 1 : remplacer uniquement l'argument `ml` de `Agent`, sans toucher à la structure `Patient`.

## Limites

Les étiquettes, la fréquence des symptômes et les seuils sont synthétiques. La cohorte construite autour de P1 rend sa prédiction impropre à démontrer la généralisation : P1 est seulement un scénario d'intégration. Le modèle peut surapprendre les règles qui les ont créés malgré le jeu de test et la validation croisée. Les probabilités ne sont pas des probabilités cliniques calibrées. Une validation sur données réelles anonymisées, revue éthique, calibration et supervision humaine seraient indispensables dans tout contexte réel.
