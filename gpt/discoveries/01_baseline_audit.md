# 01 - Audit de la baseline V1

## Objectif

Figer un point de comparaison reproductible avant toute recherche de features.

## Protocole

- 527 073 observations d'entrainement;
- 2 522 valeurs uniques de `TS`;
- 8 folds aleatoires sur les dates uniques;
- `random_state=0`;
- toutes les observations d'un meme `TS` restent dans le meme fold;
- regression Ridge sur la cible continue, puis prediction du signe.

Les labels de date sont anonymises. Leur partie numerique n'est donc pas
utilisee comme une chronologie.

## Modele audite

Le modele contient:

- huit rendements retardes: `RET_1`, `RET_2`, `RET_3`, `RET_4`, `RET_7`,
  `RET_8`, `RET_9`, `RET_18`;
- des effets fixes pour `ALLOCATION` et `GROUP`;
- une pente de `RET_1` propre a chaque allocation;
- une regularisation Ridge `alpha=100`.

## Resultats reproduits

| Mesure | Valeur |
|---|---:|
| Accuracy OOF | 0.524709 |
| Erreur standard par fold | 0.001375 |
| Score penalise par fold | 0.523334 |
| Erreur standard par TS | 0.001734 |
| Score penalise par TS | 0.522975 |
| Erreur standard par allocation | 0.001414 |
| Score penalise par allocation | 0.523295 |

La baseline naive `sign(RET_1)` atteint 0.518924. La Ridge apporte donc environ
0.58 point d'accuracy absolue au-dessus de ce signal simple.

## Verification

La reproduction a ete effectuee avec:

```bash
python gpt/model.py --n-splits 8 --random-state 0 \
  --output-dir gpt/outputs/v1_recheck
```

Le resultat est strictement identique au precedent export. La baseline peut
donc servir de reference pour les experiences suivantes.

## Risques identifies

1. La selection actuelle de huit retards est manuelle.
2. La Ridge exploite peu les volumes et le turnover.
3. Les effets de regime communs aux allocations d'une meme date sont presque
   absents.
4. Une importance de coefficient brute n'est pas comparable entre one-hot et
   variables continues; l'importance par contribution doit rester la mesure de
   lecture privilegiee.

## Decision

Conserver la V1 intacte et construire une V2 dans de nouveaux fichiers. Aucun
changement de la baseline n'est necessaire.

