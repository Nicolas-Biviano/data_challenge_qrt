# Modeles GPT pour le challenge QRT

## V2 retenue

La version finale est `model_v2.py`: une regression logistique sparse avec les
memes effets fixes que la V1, regularisee avec `C=0.003`. Elle optimise
directement le signe et atteint 0.525001 d'accuracy OOF, contre 0.524709 pour la
Ridge V1.

```bash
python gpt/model_v2.py
```

Les sorties finales sont dans `gpt/outputs/v2/`. Le chemin de recherche complet
est documente dans `RESEARCH_LOG.md` et `discoveries/`. La formulation de la V2
est dans `model_v2_math.md`.

Le bilan de pause après H13 est dans
`discoveries/41_pause_wrap_up.md`. Il sépare les conclusions établies, les
pistes seulement directionnelles et les expériences à ne pas relancer.

La reprise H14–H16 teste successivement une cible continue dé-factorisée, une
hiérarchie d'allocation à trois états et un gate de transfert avant ensemble.
Les deux modèles échouent leurs gates ; aucun ensemble ni fichier de soumission
n'est produit. Voir `discoveries/42_h14_cross_sectional_target_protocol.md` à
`discoveries/46_h16_transfer_gate_audit.md`.

## Notebooks de synthèse et d'hypothèses

- `notebooks/coding_best_practices_learned.ipynb` : guide autonome des bonnes
  pratiques Python, pandas, scikit-learn, Jupyter et validation apprises pendant
  le projet ;
- `notebooks/bilan_recherche_gpt.ipynb` : synthèse générale avec intervalles
  d'incertitude par date ;
- `notebooks/H1_test_like_training.ipynb` : entraînement limité aux dates
  test-like ;
- `notebooks/H2_auc_and_thresholds.ipynb` : AUC, seuil global, oracle par date
  et seuils appris ;
- `notebooks/H3_hard_vs_easy_dates.ipynb` : analyse détaillée des distributions
  des lignes où tous les modèles échouent ou réussissent, conditionnée au signe.
- `notebooks/H4_nonlinear_group_features.ipynb` : transformations
  `tanh/arctan/arcsin`, H4-A, forward groupée H4-F et sélection atomique
  résiduelle H4-G.
- `notebooks/H5_test_sized_panels_and_confidence.ipynb` : distribution de
  validations de 120 dates appariées au test et confiance portée par
  l'amplitude du rendement prédit.
- `notebooks/H6_conditional_error_estimator.ipynb` : probabilités d'erreur
  strictement cross-fittées, calibration, abstention diagnostique et test des
  sélecteurs V1/LightGBM.
- `notebooks/H7_zero_focused_stability_selection.ipynb` : transformations qui
  dilatent la zone proche de zéro et stability selection Elastic Net imbriquée.
- `notebooks/H8_predictable_slope_inversions.ipynb` : reproductibilité
  split-half des inversions de pente par date et tentative de prédiction X-only.
- `notebooks/H9_allocation_archetypes_and_relational_regimes.ipynb` : six
  archétypes comportementaux X-only, structure relationnelle des dates et
  diagnostic d'overfit de la prédiction des régimes.
- `notebooks/H10_defactored_returns.ipynb` : décomposition leave-one-out des
  returns en marché, écart de groupe et résidu propre, avec comparaison OOF
  appariée et barres d'erreur par date.
- `notebooks/H11_conditional_allocation_response.ipynb` : états X-only et
  fonctions de réaction par `GROUP`, `ALLOCATION` et style continu, avec
  diagnostic de shrinkage et screening atomique.
- `notebooks/H12_multiseed_and_powerful_models.ipynb` : audit H11 sur cinq
  seeds, factorization machine low-rank et LightGBM conditionnel fortement
  régularisé avec profondeur réelle vérifiée.
- `notebooks/H13_paper_tuned_random_forest.ipynb` : tuning imbriqué et
  diagnostics d'une Random Forest très régularisée, guidés par la littérature.
- `notebooks/H14_H16_controlled_resumption.ipynb` : cibles continues
  dé-factorisées, hiérarchie basse dimension et audit des gates avant ensemble.

H4-A atteint 0,524618 d'accuracy globale et 0,518056 sur les dates complètes,
contre 0,525001 et 0,518438 pour V2. Le gate n'est pas passé ; H4-B à H4-E ont
été arrêtées conformément au protocole.

H4-F obtient 0,525233 avec des groupes parfois corrélés, mais l'IC du gain
contient zéro. H4-G sélectionne une colonne atomique stable par famille puis
attaque le signal résiduel ; elle obtient 0,524686. Le rang de `TURNOVER` par
date est la famille la plus reproductible, sans transformation atomique stable.
Voir `discoveries/18_h4_atomic_residual_protocol.md` et
`discoveries/19_h4_nested_selection_results.md`.

Les notebooks sont pré-exécutés et peuvent être lus directement.

H5 montre que les panels appariés à partir de `X` reproduisent mieux le
classement public que des panels uniformes, mais restent trop instables pour
tuner des écarts de quelques centièmes de point. En revanche, l'accuracy croît
avec `|score prédit|` dans les huit folds pour V1 et LightGBM. Cette amplitude
est retenue comme mesure de confiance absolue, pas comme sélecteur de modèle.
Voir `discoveries/23_test_sized_panels_results.md`.

H6 améliore la probabilité de difficulté : le compact bat le taux constant en
Brier et log-loss dans 8/8 folds et atteint environ 58,3 % d'accuracy sur les
10 % de lignes jugées les plus sûres. Il ne sait cependant pas choisir
fiablement V1 ou LightGBM sur leurs désaccords ; aucun sélecteur ne passe le
gate. Voir `discoveries/24_h6_error_estimator_protocol.md` et
`discoveries/25_h6_error_estimator_results.md`.

H7 sélectionne très régulièrement des formes zero-focused de `RET_1`, `RET_2`
et `RET_18`, mais dégrade V2 de 0,048 point et échoue également dans la zone de
faible margin. H8 montre en revanche que les pentes feature–target changent
réellement de signe selon la date : les corrélations split-half atteignent
0,76–0,79. Le profil X-only marginal actuel ne prédit pas ces inversions
(meilleure AUC 0,528). Voir `discoveries/28_h7_zero_focused_stability_results.md`
et `discoveries/29_h8_slope_inversion_results.md`.

H9 construit six archétypes d'allocations stables entre folds (ARI 0,878) et
distincts des `GROUP` fournis (AMI 0,126). Le profil relationnel de 290 variables
sur-apprend cependant fortement : AUC train proche de 0,70 contre au mieux
0,5435 hors date. Aucun gate n'est passé. Voir
`discoveries/30_h9_allocation_archetypes_relational_protocol.md` et
`discoveries/31_h9_allocation_archetypes_relational_results.md`.

H10 teste directement les rendements dé-factorisés. Les résidus au marché
(0,522869) et au groupe (0,521486) perdent les huit folds face à V2 (0,525001).
Même l'ajout des résidus au brut perd 0,164 point avec un IC apparié entièrement
négatif. Les composantes communes, quoique moins volatiles que le résidu propre,
portent donc du signal utile dans notre représentation. Le gate PCA est fermé.
Voir `discoveries/32_h10_defactored_returns_protocol.md` et
`discoveries/33_h10_defactored_returns_results.md`.

H11 apprend comment chaque allocation réagit à huit états transversaux. Les
états seuls (0,524554), les réponses `GROUP` (0,523823) et le style continu
(0,524005) échouent. Les réponses propres aux 278 allocations atteignent
0,525318 contre 0,525001 pour V2, gagnent 6/8 folds et améliorent légèrement
l'AUC, mais l'IC du gain contient zéro. Un candidat correctement indexé est
produit sans remplacer la baseline. Voir
`discoveries/34_h11_conditional_allocation_response_protocol.md` et
`discoveries/35_h11_conditional_allocation_response_results.md`.

L'audit multi-seed ramène le gain moyen H11 à +0,011 point : 4/5 seeds positifs
et 25/40 folds gagnés, mais un IC contenant zéro. H12 teste ensuite davantage
de capacité. Le correcteur low-rank perd 0,57 point dès deux folds et sur-apprend
rapidement. LightGBM profondeur 3 obtient 0,523723 avec feuilles constantes et
0,524072 avec feuilles linéaires, sous V2. Aucun gate d'ensemble n'est ouvert.
Voir `discoveries/36_powerful_models_protocol.md`,
`discoveries/37_h11_multiseed_results.md` et
`discoveries/38_h12_powerful_models_results.md`.

## Recherche V3 — arbres lineaires et quantiles

La V3 compare LightGBM classique, `linear_tree=True`, des clusters d'allocation
construits uniquement avec `X` et une Quantile Regression Forest. Les feuilles
lineaires ameliorent LightGBM classique sur le bloc returns compact, mais aucun
modele ne bat la V2 sur le screening. Les gates ont donc arrete les huit folds,
Cubist et Local Linear Forest. Le compte rendu complet est dans
`discoveries/07_v3_tree_and_quantile_results.md`.

Les scripts reproductibles sont `v3_features.py`, `lightgbm_v3.py` et
`qrf_v3.py`. Leurs dependances sont listees dans `requirements-v3.txt` et ont
ete installees dans l'environnement local `.venv`.

Un diagnostic supplementaire a ensuite mesure les courbes train/validation et
la profondeur reelle des arbres. Le pilote non borne atteint une profondeur 12
et son ecart train-validation monte a 3.10 points : il est clairement en
overfit. Une grille imbriquee utilisant profondeur, feuilles, tailles minimales,
L1/L2, `linear_lambda`, `path_smooth`, gain minimal, subsampling et
regularisation categorielle corrige cet overfit. Elle obtient toutefois
0.524835 sur deux folds externes, contre 0.525112 pour la V2, et choisit une
profondeur differente sur chaque fold. Le detail est dans
`discoveries/08_lightgbm_bias_variance_diagnostic.md`.

## `SIGNED_VOLUME_1` et ses valeurs manquantes

Une representation deux-parties separe maintenant la disponibilite du volume
de sa valeur observee. Les valeurs, rangs et interactions n'ameliorent pas la
V2 globale. Un specialiste ajuste uniquement sur les 26.48 % de lignes ou le
volume est disponible atteint 0.525185 sur huit folds, contre 0.525001 pour la
V2, mais ne progresse que sur quatre folds. Un shrinkage selectionne hors fold
degrade le score. Ce signal de regime est donc conserve pour diversification,
pas comme nouveau modele final. Voir
`discoveries/09_signed_volume_two_part.md`.

Un audit des moyennes conditionnelles par quantile sur les residus OOF de la V2
a ensuite analyse 47 features non-return. Il detecte plusieurs formes
non lineaires, mais leur test huit folds ne produit au mieux que 0.525024
d'accuracy, contre 0.525001 pour V2. Les profils, leur stabilite entre folds et
les transformations testees sont documentes dans
`discoveries/10_conditional_quantile_means.md`.

## V1 — effets fixes Ridge

Cette tentative part de deux constats simples :

1. `RET_1` contient le signal directionnel individuel le plus net ;
2. le taux de cible positive varie durablement entre les allocations.

Le modèle prédit d'abord le rendement continu avec une régression Ridge, puis
seuille la prédiction à zéro. La matrice contient :

- les effets fixes `ALLOCATION` et `GROUP` encodés en one-hot ;
- `RET_1`, `RET_2`, `RET_3`, `RET_4`, `RET_7`, `RET_8`, `RET_9`, `RET_18` ;
- une interaction `ALLOCATION × RET_1`, qui permet une sensibilité différente
  au rendement le plus récent selon l'allocation.

La régularisation ramène les effets et pentes peu documentés vers l'effet
commun. L'imputation, la standardisation et les encodages sont réajustés dans
chaque fold. La validation passe directement par le framework existant
`src.cross_validation.run_cv`, avec des dates entières affectées à chaque fold.

## Exécution

Depuis la racine du dépôt :

```bash
python gpt/model.py
```

Pour un test plus rapide :

```bash
python gpt/model.py --n-splits 4
```

Les fichiers produits dans `gpt/outputs/` sont :

- `cv_summary.json` : accuracy, erreur standard et score pénalisé par fold,
  date et allocation, avec le benchmark naïf `sign(RET_1)` ;
- `oof_predictions.csv` : prédictions out-of-fold utilisées pour analyser la
  stabilité par date et par allocation ;
- `fold_metrics.csv` : métriques détaillées par fold ;
- `submission.csv` : fichier au format de `sample_submission.csv` ;
- `test_predictions.csv` : classe et score continu pour diagnostic.

Cette première version privilégie une hypothèse interprétable et une matrice
sparse légère. Elle ne suppose aucun ordre chronologique des dates anonymisées.

La formulation mathématique du modèle est disponible dans `model_math.md`.

## Résultat obtenu

Exécution du 2 août 2026, avec 8 folds et `random_state=0` :

| Modèle | Accuracy | Score pénalisé par fold | Score pénalisé par date |
|---|---:|---:|---:|
| Ridge à effets fixes | 0,52471 | 0,52333 | 0,52297 |
| Baseline `sign(RET_1)` | 0,51892 | 0,51788 | 0,51716 |

La soumission produite contient 31 870 lignes et 55,16 % de prédictions
positives. Le détail machine-readable est conservé dans
`outputs/cv_summary.json`.
