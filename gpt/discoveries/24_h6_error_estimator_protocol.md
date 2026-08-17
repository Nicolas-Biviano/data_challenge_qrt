# H6 — protocole de l'estimateur d'erreur conditionnelle

## Hypothèse

L'amplitude `|score|` distingue bien les lignes faciles des lignes difficiles,
mais ne choisit pas correctement entre V1 et LightGBM lorsqu'ils divergent. H6
teste si un estimateur d'erreur conditionnelle peut faire mieux en combinant
les deux scores, un petit ensemble de caractéristiques X-only et le régime de
la date.

## Interdictions

- aucun target encoding ;
- aucune information du test public ou du leaderboard dans l'entraînement ;
- aucune feature calculée avec `target` ;
- aucun tuning de seuil sur les prédictions H6 ;
- aucune soumission produite par cette expérience.

## Cross-fitting

Les prédictions V1 et LightGBM utilisées comme features sont déjà OOF. Pour
chaque fold externe H6, l'estimateur d'erreur est entraîné sur les sept autres
folds et prédit le fold tenu à l'écart. Une date complète reste donc toujours
d'un seul côté de la séparation.

Cette seconde couche de cross-fitting est indispensable : ajuster et évaluer
l'estimateur d'erreur sur les mêmes lignes donnerait une mesure optimiste.

## Cibles

Deux cibles binaires sont estimées séparément :

- `error_v1 = 1[V1 se trompe]` ;
- `error_lgbm = 1[LightGBM se trompe]`.

Sur les seules lignes où les signes divergent, une troisième cible vaut 1 si
LightGBM est correct. Comme le problème est binaire, exactement un des deux
modèles est alors correct.

## Features autorisées

1. Scores OOF : scores signés, amplitudes, moyenne, différence, produit,
   minimum/maximum d'amplitude et désaccord.
2. Confiance relative dans la date : rang percentile de chaque amplitude.
3. Returns compacts : lags 1, 2, 3, 4, 7, 8, 9 et 18, leurs amplitudes, moments
   récents et globaux, changements de signe.
4. Liquidité : disponibilité et amplitude de `SIGNED_VOLUME_1`, fraction de
   volumes observés et turnover.
5. Régime X-only de la date : taille, moyenne/dispersion de `RET_1`, part de
   `RET_1` positifs, missingness du volume et rangs intra-date.
6. `ALLOCATION` et `GROUP` en one-hot uniquement.

## Estimateurs pré-enregistrés

- **constant** : taux d'erreur du train externe ;
- **amplitude_logit** : régression logistique sur l'amplitude et son rang dans
  la date, `C=0.01` ;
- **compact_logit** : régression logistique fortement régularisée sur toutes
  les features autorisées, `C=0.01`.

Les numériques sont imputées par médiane et standardisées dans chaque fold.
Les catégories sont imputées puis one-hot encodées dans chaque fold. Aucun
hyperparamètre n'est sélectionné après lecture des résultats.

## Critères principaux

Pour la probabilité d'erreur : Brier, log-loss, ROC-AUC, calibration par décile
et écart train-validation. Un modèle n'est utile que s'il bat le taux constant
en Brier et log-loss de façon stable sur les folds.

Pour la sélection de modèle :

- accuracy globale ;
- accuracy sur les désaccords ;
- gain apparié à V1 par fold ;
- distribution du gain sur les 1 000 panels H5 appariés.

Le sélecteur ne passe le gate que si son gain sur V1 est positif dans au moins
7 folds sur 8 et dans au moins 90 % des panels appariés. Ce gate volontairement
strict tient compte de l'échec public du blend.

