# 08 - Diagnostic biais/variance des arbres lineaires

## Question

Le LightGBM `linear_tree=True` precedent etait-il reellement trop regularise,
ou au contraire en overfit ? Quelle profondeur est suffisamment fiable pour
ce jeu de donnees ?

Le diagnostic est reproductible avec :

```bash
.venv/bin/python gpt/lightgbm_diagnostics.py --outer-fold 1
.venv/bin/python gpt/lightgbm_diagnostics.py --outer-fold 2
```

## Protocole sans fuite

Pour chacun des deux folds externes :

1. le fold externe est mis de cote et n'intervient dans aucun choix ;
2. les dates du train externe sont redivisees en train/validation interne ;
3. neuf configurations sont comparees sur la validation interne ;
4. la configuration et le nombre d'arbres sont choisis uniquement en interne ;
5. ce choix est reentraine sur tout le train externe, puis evalue une seule
   fois sur le fold externe intact.

Les labels de date ne sont pas tries et aucun target encoding n'est utilise.
Les clusters d'allocation sont appris uniquement avec `X` du train.

## Regularisations explorees

La grille ne se limite pas a `max_depth`. Elle contraint conjointement :

- profondeur maximale : 2, 3, 4 ou 5 ;
- nombre maximal de feuilles, volontairement inferieur a `2^depth` dans les
  variantes fortes ;
- nombre minimal d'observations et somme minimale des Hessiennes par feuille ;
- penalites L1 et L2 des arbres ;
- `linear_lambda` pour les regressions locales dans les feuilles ;
- `path_smooth` pour lisser un enfant vers son parent ;
- gain minimal necessaire pour autoriser un split ;
- sous-echantillonnage des lignes, des colonnes par arbre et par noeud ;
- `cat_l2`, `cat_smooth`, taille minimale d'un groupe categoriel et nombre de
  seuils categoriels testes ;
- nombre d'arbres, choisi sur les courbes 10, 25, 50, 100, 150, 250 et 400.

Les penalites de feuille sont calibrees avec la taille du train et non choisies
comme de petits nombres arbitraires. Les gains minimaux proviennent des
quantiles de gains observes dans un pilote.

## Ce que montre le modele courant

Le pilote avec 16 feuilles et sans limite explicite de profondeur n'est pas
sous-ajuste : il sur-apprend nettement.

Sur le premier split interne :

| Arbres | Accuracy train | Accuracy validation | Ecart |
|---:|---:|---:|---:|
| 10 | 0.52704 | 0.52177 | +0.00527 |
| 25 | 0.53025 | 0.52234 | +0.00791 |
| 100 | 0.53804 | 0.52215 | +0.01589 |
| 400 | 0.55294 | 0.52190 | +0.03104 |

Ses 400 arbres ont une profondeur reelle moyenne de 7.41 et atteignent 12,
malgre seulement 16 feuilles par arbre. La feuille au quantile 5 % ne contient
qu'environ 1 188 observations. Le probleme vient donc bien de branches trop
longues et de regressions locales trop flexibles, pas d'un manque de capacite.

## Effet de la profondeur et de la force de regularisation

Sur le premier split interne, les meilleures accuracies par configuration
sont :

| Configuration | Arbres retenus | Train | Validation | Ecart |
|---|---:|---:|---:|---:|
| profondeur 4, forte | 25 | 0.52454 | **0.52605** | -0.00151 |
| profondeur 4, moderee | 100 | 0.52829 | 0.52565 | +0.00264 |
| profondeur 5, moderee | 150 | 0.53519 | 0.52542 | +0.00977 |
| profondeur 3, moderee | 250 | 0.52968 | 0.52510 | +0.00458 |
| pilote courant | 25 | 0.53025 | 0.52234 | +0.00791 |

La profondeur 4 forte atteint en pratique 3.26 niveaux en moyenne, avec 5.60
feuilles par arbre et une feuille mediane de 15 074 lignes. A l'inverse, la
profondeur 5 moderee utilise ses 5 niveaux, descend a une feuille mediane de
1 798 lignes et recommence a sur-apprendre apres 150 arbres.

Le petit ecart train-validation negatif a 25 arbres est compatible avec le
sous-echantillon aleatoire de 100 000 lignes utilise pour mesurer le train ; ce
n'est pas une preuve que la validation est plus facile.

## Validation externe imbriquee

| Fold externe | Choix interne | Accuracy LightGBM | Logistique V2 | Ancien LightGBM |
|---:|---|---:|---:|---:|
| 1 | profondeur 4 forte, 25 arbres | 0.522849 | **0.525077** | 0.524789 |
| 2 | profondeur 5 forte, 50 arbres | **0.526820** | 0.525147 | 0.525238 |
| moyenne | choix imbrique par fold | 0.524835 | **0.525112** | 0.525013 |

Le choix de profondeur n'est pas stable entre les deux validations internes et
le gain du second fold ne compense pas la perte du premier. La moyenne est
legerement inferieure a la V2 (-0.000277), avec une variance de gain beaucoup
trop grande pour annoncer une amelioration.

## Decision

La regularisation forte corrige effectivement l'overfit du LightGBM courant.
Elle transforme un modele manifestement trop flexible en candidat plausible,
mais elle ne fournit pas une profondeur fiable ni un gain externe stable.

Il ne serait pas rationnel de lancer les six folds restants : le gate sur deux
folds externes est manque. La V2 logistique reste le modele principal. Si cette
branche est reprise plus tard, la seule zone raisonnable est profondeur 4 ou 5,
forte regularisation et early stopping avant 100 arbres ; les arbres profonds
sans limite sont exclus.

## Fichiers produits

- `gpt/lightgbm_diagnostics.py` : protocole imbrique et audit des structures ;
- `gpt/outputs/v3_lgbm_diagnostics/curves_outer_*.csv` : courbes
  train/validation ;
- `structures_outer_*.csv` : profondeurs, feuilles, gains et normes des
  coefficients locaux ;
- `best_by_config_outer_*.csv` : meilleur nombre d'arbres par configuration ;
- `selection_outer_*.json` : choix interne et score externe intact.
