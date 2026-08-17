# 03 - Calibrage de la branche Ridge

## Objectif

Verifier si le score de la V1 peut etre ameliore sans changer ses variables,
uniquement par regularisation ou calibration du seuil.

## Regularisation globale

| Alpha | Accuracy | Score penalise TS |
|---:|---:|---:|
| 10 | 0.524620 | 0.522891 |
| 30 | 0.524681 | 0.522951 |
| 100 | **0.524709** | **0.522975** |
| 300 | 0.524686 | 0.522935 |
| 1000 | 0.524369 | 0.522582 |

La courbe est plate entre 30 et 300. `alpha=100` reste le meilleur choix sur
l'accuracy et le score penalise.

## Penalisation des pentes par allocation

Multiplier les colonnes `ALLOCATION x RET_1` change leur penalisation relative
dans la Ridge.

| Echelle de l'interaction | Accuracy | Score penalise TS |
|---:|---:|---:|
| 0.25 | 0.524626 | 0.522859 |
| 0.50 | 0.524673 | 0.522926 |
| 1.00 | **0.524709** | **0.522975** |
| 2.00 | 0.524696 | 0.522966 |
| 4.00 | 0.524694 | 0.522965 |

Les echelles 1, 2 et 4 sont presque equivalentes. L'echelle historique 1 est
conservee pour eviter un reglage sans gain observable.

## Calibration du seuil

Sur l'ensemble des predictions OOF, le meilleur seuil apparent apporte environ
0.022 point. Ce resultat est optimiste parce que le meme OOF sert a choisir et
a mesurer le seuil.

Un test honnete a donc ete effectue: pour chaque fold, le seuil est choisi sur
les sept autres folds puis applique au fold exclu. L'accuracy obtenue est
0.524140, contre 0.524709 avec le seuil naturel zero.

## Decision

- conserver `alpha=100`;
- conserver l'echelle d'interaction 1;
- conserver le seuil zero;
- chercher le prochain gain dans une autre famille de modeles;
- ne pas utiliser de target encoding.

## Reproductibilite

La grille est implementee dans `gpt/ridge_calibration.py`; les sorties se
trouvent dans `gpt/outputs/ridge_alpha/` et
`gpt/outputs/ridge_interaction/`.

