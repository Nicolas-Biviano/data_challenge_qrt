# 05 - Decision finale V2

## Selection ciblee et volumes manquants

Le screening des residus a fait ressortir la dispersion de `SIGNED_VOLUME_1`
dans `TS x GROUP`. L'audit a ensuite montre:

- 73.52 % de valeurs manquantes dans le train;
- 75.04 % dans le test;
- generalement 6 valeurs observees par `TS x GROUP`, parfois 65 a 70;
- un schema de couverture proche entre train et test.

La Ridge atteint 0.524935 avec la dispersion de `SIGNED_VOLUME_1`, 0.525045
avec le nombre de volumes observes par date et 0.525090 avec count SV1 et
dispersion SV2. Ces gains semblent donc partiellement lies au regime de
missingness. Ils ne sont pas retenus dans la V2 principale afin de limiter la
dependance a un artefact de donnees.

## Test final d'ensemble

Un ensemble Ridge + logistique a ete calibre hors fold. Pour chaque fold, le
poids est choisi sur les sept autres folds. Les poids Ridge choisis vont de
0.025 a 0.9, signe d'une forte instabilite. Le resultat final est:

| Modele | Accuracy | Score penalise TS |
|---|---:|---:|
| Ridge V1 | 0.524709 | 0.522975 |
| Logistique V2 | **0.525001** | **0.523285** |
| Ensemble hors fold | 0.524426 | 0.522706 |

L'ensemble est rejete.

## Modele retenu

La V2 est une regression logistique L2 avec `C=0.003`, ajustee sur:

- `RET_1`, `RET_2`, `RET_3`, `RET_4`, `RET_7`, `RET_8`, `RET_9`, `RET_18`;
- one-hot `ALLOCATION` et `GROUP`;
- interactions `ALLOCATION x RET_1`.

Elle n'utilise ni ordre de date, ni target encoding, ni feature issue de la
reconstruction des fenetres.

## Resultats et soumission

| Mesure | Valeur |
|---|---:|
| Accuracy OOF | 0.525001 |
| Erreur standard par TS | 0.001716 |
| Score penalise TS | 0.523285 |
| Erreur standard par allocation | 0.001353 |
| Lignes test | 31 870 |
| Taux de predictions positives test | 0.602259 |

Le taux positif du test est plus eleve que celui de la V1. Aucun recalibrage de
seuil n'est applique, car le calibrage hors fold teste precedemment degradait
l'accuracy.

## Artefacts

- `gpt/model_v2.py`: entrainement complet et soumission;
- `gpt/model_v2_math.md`: formulation mathematique;
- `gpt/model_v2_presentation.ipynb`: presentation et diagnostics;
- `gpt/outputs/v2/submission.csv`: soumission finale;
- `gpt/outputs/v2/summary.json`: resume machine-readable.

