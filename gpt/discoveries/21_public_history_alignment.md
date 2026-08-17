# Historique public et alignement des validations

## Treize soumissions disponibles

L'historique communiqué le 5 août 2026 est conservé dans
`gpt/outputs/leaderboard_alignment/public_history.csv`. Les meilleurs scores
sont :

1. LightGBM à feuilles linéaires : 0,512018 ;
2. Ridge à effets fixes : 0,511829 ;
3. logistique ElasticNet `C=0.01, l1_ratio=0.7` : 0,511453.

Le LightGBM GBDT à feuilles constantes tombe à 0,499843 et la Random Forest à
0,507248. Les deux meilleurs modèles régressent la cible continue avant de
prendre son signe. Cela favorise pour la suite les modèles linéaires ou
localement linéaires entraînés sur la magnitude, plutôt que la classification
directe ou les feuilles constantes.

Deux lignes ElasticNet affichent les mêmes paramètres `C=0.001,
l1_ratio=0.4` mais diffèrent de 0,00546 sur le public. Le feature set,
preprocessing ou fichier final différait donc nécessairement. On ne peut pas
relier honnêtement les onze anciens runs à une CV locale sur la seule base de
leur nom et de leurs hyperparamètres.

## Trois familles de modèles reliables

| Modèle | CV ordinaire | Holdout test-like | Public |
|---|---:|---:|---:|
| V1 Ridge | 0,524709 | 0,533970 | 0,511829 |
| V2 locale `C=0.003` / logistique publique `C=0.001` | 0,525001 | 0,533610 | 0,507060 |
| LightGBM linéaire | 0,524442 | 0,531492 | 0,512018 |

La correspondance de la ligne logistique est approximative car le `C` public
indiqué diffère du `C=0.003` local. Sur ces trois familles, la corrélation de
rang de Spearman entre CV ordinaire et public vaut néanmoins `−1,0`.
Entre holdout adversarial test-like et public, elle vaut encore `−0,5`. Le
holdout test-like classe correctement V1 devant V2 mais place LightGBM dernier,
alors qu'il est premier public. Aucun de ces estimateurs ne doit donc servir au
tuning fin.

## Probe suivante

V1 et LightGBM sont les deux meilleurs modèles publics et leurs prédictions OOF
diffèrent sur 16,0 % des lignes. Un blend préfixé 50/50 de leurs scores continus
atteint 0,525202 OOF, contre 0,524709 et 0,524442 séparément. Le poids n'a pas
été choisi avec le leaderboard.

Ce blend est la prochaine soumission la plus informative. S'il ne dépasse pas
les composants, il faudra cesser d'utiliser le leaderboard pour choisir des
poids et revenir à la recherche de signal continu transférable.

## Résultat du blend

Le blend 50/50 obtient **0,510700** public, contre 0,511829 pour V1 et
0,512018 pour LightGBM. Il perd donc respectivement 0,001130 et 0,001318,
malgré son gain OOF de 0,000493 face à V1.

Le gain d'ensemble OOF ne se transfère pas. Aucun nouveau poids de blend ne
sera testé : choisir 25/75 ou 75/25 après ce résultat serait du tuning sur le
leaderboard. Les prochaines soumissions doivent départager des hypothèses
structurelles, en priorité la présence ou l'absence des effets catégoriels.
