# H4 — résultats d'exécution

## Étape exécutée

H4-A ajoute 48 features X-only non linéaires au socle V2 : z-scores robustes,
rangs `TS` et `TS × GROUP`, `tanh`, `arctan`, signed-log, hinges, disponibilité
de SV1 et part de volumes positifs. La matrice comporte 616 colonnes sparse.

Le modèle conserve `C=0.003`, les mêmes huit folds de dates et les mêmes effets
fixes que V2.

## Résultats

| Mesure | V2 | H4-A | Différence |
|---|---:|---:|---:|
| Accuracy globale | 0,525001 | 0,524618 | -0,000383 |
| Accuracy dates complètes | 0,518438 | 0,518056 | -0,000381 |
| AUC | 0,535216 | 0,535543 | +0,000326 |
| Balanced accuracy | 0,523483 | 0,523406 | -0,000077 |
| Accuracy y=0 | 0,417790 | 0,439034 | +0,021244 |
| Accuracy y=1 | 0,629175 | 0,607778 | -0,021397 |
| Taux positif prédit | 0,606030 | 0,584708 | -0,021322 |

La différence appariée globale a un IC bootstrap à 95 % de
[-0,001582 ; 0,000814]. Sur les dates complètes, l'IC vaut
[-0,001935 ; 0,001110].

## Interprétation

H4-A ordonne marginalement mieux les observations, mais déplace la calibration
vers moins de positifs. Le gain de 2,12 points sur les vrais négatifs est
presque exactement compensé par une perte de 2,14 points sur les vrais
positifs. Le balanced accuracy reste pratiquement inchangé.

## Gate

Le protocole exigeait un gain d'accuracy global et sur les dates complètes.
H4-A est négative sur les deux métriques. Les étapes H4-B à H4-E n'ont donc pas
été exécutées, conformément au plan préenregistré.

## Décision

H4-A est rejetée comme remplacement de V2. Les transformations saturantes sont
utiles pour comprendre le biais positif, mais leur ajout simultané ne produit
pas de gain d'accuracy validé.
