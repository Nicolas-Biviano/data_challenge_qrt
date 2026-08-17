# 04 - Comparaison de modeles et regimes

## Contraintes

- aucun target encoding;
- aucun ordre suppose entre les dates;
- categories encodees uniquement par one-hot independant de `y`;
- tous les scores complets utilisent les huit folds fixes de `TS`.

## Boosting au niveau ligne

Un HistGradientBoosting regularise a ete teste sur 97 variables numeriques:
resumes de rendements, liquidite, rangs par date et statistiques par groupe.

| Modele | Folds de screening | Accuracy |
|---|---:|---:|
| HGB regression cible continue | 2 | 0.519876 |
| HGB classification directe | 2 | 0.522656 |

Les resultats sont sous la Ridge sur les memes folds. Le calcul complet n'est
donc pas justifie.

## Classification logistique sparse

La logistique utilise exactement le design de la V1:

- huit retards selectionnes;
- one-hot `ALLOCATION` et `GROUP`;
- interactions `ALLOCATION x RET_1`;
- aucun encodage base sur la cible.

| C | Accuracy sur 8 folds | Score penalise TS |
|---:|---:|---:|
| 0.001 | 0.524561 | 0.522784 |
| 0.003 | **0.525001** | **0.523285** |
| Ridge V1 | 0.524709 | 0.522975 |

La logistique `C=0.003` gagne 0.029 point d'accuracy et 0.031 point de score
penalise par date. Elle gagne sur cinq folds sur huit.

Le test exact sur les predictions discordantes donne cependant `p=0.271`. Ce
gain est donc un candidat utile, pas encore une amelioration statistiquement
etablie. Sur les six folds qui n'ont pas servi au screening initial, son delta
reste positif a +0.017 point.

## Modele de regime par date

Une table de 2 522 dates est construite avec moyennes, dispersions et proportions
de signes positifs de `X`. Une seconde table contient 7 665 couples
`TS x GROUP`, joints aux statistiques globales de leur date.

La cible du modele est soit le rendement futur moyen, soit le taux futur de
signes positifs. La cible n'est jamais utilisee comme variable d'entree et les
dates de validation sont absentes du train.

| Niveau et modele | Accuracy ligne | Score penalise TS |
|---|---:|---:|
| Date Ridge, taux positif | 0.506713 | 0.504802 |
| Date HGB, taux positif | 0.505795 | 0.503883 |
| Date-groupe HGB, rendement | 0.508148 | 0.506305 |
| Date-groupe ExtraTrees, taux positif | **0.511774** | **0.509916** |

## Decouvertes

1. La non-linearite seule ne remplace pas les effets fixes de la V1.
2. Optimiser directement le signe avec une logistique apporte un petit gain.
3. Le taux positif d'une date varie fortement, mais les agregats simples de son
   historique ne permettent pas de predire seuls ce regime.
4. Le modele date-groupe ExtraTrees est trop faible pour etre utilise seul. Ses
   erreurs pourront seulement etre etudiees lors de la phase d'ensemble.

## Decision

- conserver la logistique `C=0.003` comme premier candidat V2;
- rejeter les HGB au niveau ligne;
- rejeter les modeles de regime comme modeles autonomes;
- chercher maintenant quelques features individuelles complementaires, sans
  empiler des familles entieres;
- reporter ensemble et stacking a la derniere phase.

