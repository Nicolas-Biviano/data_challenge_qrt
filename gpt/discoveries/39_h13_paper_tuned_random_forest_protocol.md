# H13 — Random Forest régularisée selon la littérature

## Sources primaires

- Probst, Wright et Boulesteix, [Hyperparameters and Tuning Strategies for
  Random Forest](https://arxiv.org/abs/1804.03515) : `mtry`, fraction
  d'échantillonnage et taille des nœuds contrôlent le compromis force/corrélation
  des arbres ; `mtry` est généralement le paramètre le plus important à tuner.
- Mentch et Zhou, [Randomization as Regularization](https://www.jmlr.org/papers/v21/19-905.html) :
  la randomisation des variables joue un rôle analogue à une pénalisation dans
  les contextes à faible signal/bruit.
- Probst et Boulesteix, [To tune or not to tune the number of
  trees](https://arxiv.org/abs/1705.05654) : ne pas sélectionner opportunément
  le nombre d'arbres ; choisir un nombre computationnellement suffisant et
  contrôler la convergence.
- Breiman, [Random Forests](https://doi.org/10.1023/A:1010933404324) : la
  généralisation dépend de la force des arbres et de leur corrélation ; les
  estimations out-of-bag permettent un diagnostic interne.

## Représentation sans target encoding

La forêt reçoit :

- les huit returns de V2 ;
- les huit états H11 leave-one-out ;
- les résidus `RET_1` au marché et au groupe ;
- huit profils continus d'allocation, recalculés uniquement sur les dates
  d'entraînement du fold ;
- le `GROUP` en one-hot.

L'identité `ALLOCATION` n'est pas injectée comme entier ordinal. Les profils
X-only permettent à la forêt d'apprendre des interactions de style sans ordre
artificiel et sans target encoding.

## Sélection imbriquée

Pour chaque fold externe intact :

1. les dates d'entraînement externes sont séparées en train/validation internes ;
2. quatre configurations de 300 arbres sont comparées sur la validation interne ;
3. la meilleure est réajustée sur toutes les dates d'entraînement externes ;
4. le nombre d'arbres final est fixé à 1 000, sans sélection sur le fold externe ;
5. les performances à 100/300/600/1 000 arbres sont seulement une courbe de
   convergence diagnostique.

## Espace de régularisation

| Configuration | profondeur | feuille minimale | max features | max samples | feuilles max |
|---|---:|---:|---:|---:|---:|
| ultra random | 3 | 2 % | 25 % | 40 % | 8 |
| very strong | 4 | 1 % | 40 % | 50 % | 12 |
| strong | 5 | 0,5 % | 60 % | 63,2 % | 20 |
| pruned | 6 | 0,25 % | 35 % | 50 % | 24 |

Les profondeurs, nombres de feuilles, tailles réelles des feuilles, train/OOB et
validation externe sont tous enregistrés. Deux folds externes servent de
screening ; les huit folds ne sont autorisés que si la forêt bat V2 sur la
moyenne et ne présente pas un écart train-validation excessif.
