# H3 approfondie — distributions des lignes unanimement fausses

## Population

Les cinq fichiers OOF sont exactement alignés sur 527 073 `ROW_ID` :

- 199 707 lignes tous justes ;
- 174 050 lignes tous faux ;
- 153 316 lignes mixtes.

Les cas extrêmes représentent 70,91 % du train. Le taux positif vaut 61,89 %
sur les lignes tous justes et 38,37 % sur les lignes tous faux. Toute analyse
non conditionnée à la cible mélange donc difficulté et biais de signe.

## Mesures de distribution

Pour chaque variable numérique brute ou X-only dérivée, l'analyse calcule :

- différence standardisée de moyenne ;
- différence robuste de médiane normalisée par l'IQR ;
- distance KS ;
- PSI ;
- Wasserstein approximée sur neuf quantiles ;
- différence de missingness ;
- intervalle à 95 % du SMD par bootstrap de dates pour les vingt premiers
  effets globaux.

Globalement, les différences numériques sont minuscules : le SMD maximal est
d'environ 0,06. Plusieurs intervalles excluent zéro grâce à la taille de
l'échantillon, sans que leur taille d'effet devienne utile.

## Annulation conditionnelle au signe

À cible fixée, les distributions changent fortement :

- pour `y=0`, le SMD de `RET_1` vaut environ +1,55 et KS 0,677 ;
- pour `y=1`, le SMD vaut environ -1,52 et KS 0,670 ;
- le rang transversal de `RET_1` atteint environ ±1,9 SMD.

Les signes s'inversent presque parfaitement. Le phénomène décrit le conflit
entre momentum récent et réalisation future ; il ne donne pas une règle
déployable puisque `y` est inconnu.

## `SIGNED_VOLUME_1`

La valeur est absente sur environ trois quarts des lignes extrêmes. La
missingness des lignes tous faux moins tous justes vaut :

- +1,06 point globalement ;
- -5,07 points lorsque `y=0` ;
- +6,79 points lorsque `y=1`.

L'effet global faible cache donc une interaction forte mais opposée avec le
signe futur.

## Allocations

`ALLOCATION` domine l'importance du détecteur d'erreur. Pour certaines
allocations, les cinq modèles produisent presque toujours un signe constant.
`ALLOCATION_152`, par exemple, a un taux positif réel OOF de 65,89 %, tandis
que V2 prédit positif dans 99,84 % des cas. Son accuracy élevée reflète surtout
le déséquilibre de classe.

Parmi 275 allocations ayant au moins 300 observations extrêmes dans chaque
classe, seulement 12 conservent un effet de difficulté de même direction pour
`y=0` et `y=1`.

## Test hors échantillon

Les détecteurs LightGBM sont fortement régularisés et validés en cinq folds de
dates entières, sans target encoding :

| Bloc | AUC globale | AUC y=0 | AUC y=1 |
|---|---:|---:|---:|
| Returns | 0,510 | 0,460 | 0,558 |
| Volumes | 0,504 | 0,478 | 0,528 |
| Structure + catégories | 0,523 | 0,478 | 0,567 |
| Toutes X enrichies | 0,524 | 0,466 | 0,580 |

Le comportement opposé par classe invalide l'idée d'un détecteur universel de
lignes difficiles.

## Hypothèse suivante

Les nouvelles transformations non linéaires ne font pas partie de H3. Elles
sont isolées dans **H4**, documentée dans
`gpt/discoveries/16_h4_nonlinear_group_features.md` et dans son notebook dédié.

Notebook : `gpt/notebooks/H3_hard_vs_easy_dates.ipynb`.
