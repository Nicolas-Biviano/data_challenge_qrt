# H11 — résultats des fonctions de réaction conditionnelles

## Résultat principal

L'intuition est partiellement confirmée : l'état de marché n'améliore pas V2
par lui-même, mais permettre à chaque allocation d'y répondre différemment
produit le seul gain positif de H11.

| Modèle | Accuracy | Gain vs V2 | IC 95 % apparié | Folds gagnés |
|---|---:|---:|---:|---:|
| V2 | 0,525001 | 0 | — | — |
| États seuls | 0,524554 | -0,000448 | [-0,001384 ; 0,000489] | 1/8 |
| `GROUP × états` | 0,523823 | -0,001178 | [-0,002470 ; 0,000114] | 4/8 |
| `ALLOCATION × états`, échelle 0,20 | **0,525318** | **+0,000317** | [-0,000648 ; 0,001282] | **6/8** |
| `GROUP + ALLOCATION × états` | 0,524018 | -0,000983 | [-0,002268 ; 0,000303] | 2/8 |
| Style continu × états | 0,524005 | -0,000996 | [-0,002408 ; 0,000415] | 3/8 |

L'AUC moyenne passe de 0,535416 à 0,535727 pour la réponse par allocation. Le
gain d'accuracy est cohérent dans six folds, mais son intervalle contient zéro.
Il s'agit donc d'un candidat diversifiant, pas d'une nouvelle baseline prouvée.

## Interprétation

Les moyennes de marché ne suffisent pas à prévoir la target. Les interactions
par `GROUP` ne suffisent pas non plus. En revanche, l'identité détaillée de
l'allocation contient une information sur sa réponse conditionnelle.

Cela correspond à l'intuition économique : deux allocations du même groupe
peuvent avoir des expositions, horizons et règles de rééquilibrage différents.
`GROUP` est trop grossier pour capturer cette hétérogénéité.

## Screening atomique

Les huit états ont ensuite été testés un par un avec une pente par allocation.
Ce screening est exploratoire : sélectionner son meilleur état sur les mêmes
OOF créerait un biais de sélection.

- dispersion de `RET_1` : +0,000070, 4/8 folds ;
- momentum de marché court : +0,000049, 6/8 folds ;
- tous les autres sont négatifs ou quasi nuls ;
- volatilité récente : -0,000389 avec IC entièrement négatif.

Aucun état isolé ne reproduit le gain du bloc complet. Le signal semble venir
d'une configuration multivariée plutôt que d'un déclencheur unique.

## Courbe de shrinkage

| Échelle des interactions | Accuracy | Folds gagnés | Accuracy train moyenne |
|---:|---:|---:|---:|
| 0,10 | 0,524806 | 4/8 | 0,530281 |
| 0,20 | **0,525318** | **6/8** | 0,531478 |
| 0,35 | 0,525314 | 5/8 | 0,533996 |

Une échelle 0,10 sous-ajuste. Passer de 0,20 à 0,35 n'améliore plus la validation
mais augmente fortement le score train : le modèle commence à mémoriser les
réponses d'allocation. L'échelle préenregistrée 0,20 est donc conservée ; le
point 0,35 n'est pas retenu après coup.

## Peut-on résumer l'identité par un style ?

Le modèle low-rank fondé sur huit profils X-only — moyenne et volatilité de
`RET_1`, autocorrélation, exposition au marché, exposition au groupe, momentum
et turnover — tombe à 0,524005. Ces descriptions sont stables mais ne résument
pas les différences de réaction utiles.

La matrice des coefficients apprise par allocation est pourtant structurée :
ses trois premiers axes SVD expliquent 72,9 % de sa variance. Cela suggère des
styles latents, mais ils ne coïncident pas avec nos statistiques marginales
simples. Apprendre ces axes avec la target exigerait une procédure imbriquée ;
on ne doit pas les réinjecter directement après une SVD full-data.

## Candidat test

Le modèle préenregistré à l'échelle 0,20 a été ajusté sur tout le train. Le CSV
de soumission respecte exactement l'index test et prédit 60,36 % de positifs.
Il ne corrige donc pas le biais de classe observé sur les modèles précédents.

## Décision

H11 est la première piste récente qui soutient l'idée de réponse conditionnelle
au niveau allocation. Elle mérite un probe public si une soumission est
disponible, mais le gain OOF de 0,032 point reste inférieur à son incertitude.
On ne remplace pas V2 avant confirmation externe.

La suite rigoureuse serait une factorisation hiérarchique/low-rank apprise dans
chaque fold externe, avec très peu de dimensions. Elle ne doit être lancée que
si le probe public H11 n'est pas manifestement inférieur à V1/V2.
