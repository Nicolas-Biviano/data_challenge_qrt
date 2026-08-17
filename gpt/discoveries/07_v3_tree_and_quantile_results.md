# 07 - Resultats V3: arbres lineaires, categories X-only et QRF

## Reference sur les deux folds de screening

La logistique V2 atteint environ 0.52511 sur les deux premiers folds. Les
experiences V3 doivent la battre sur exactement les memes observations avant
de passer aux huit folds.

## LightGBM: feuilles constantes contre lineaires

### Bloc returns compact

| Modele | Accuracy 2 folds | Score penalise TS |
|---|---:|---:|
| LightGBM constant | 0.524190 | 0.520959 |
| LightGBM `linear_tree=True` | **0.525014** | **0.521876** |
| Logistique V2, memes folds | environ **0.52511** | - |

Les feuilles lineaires gagnent 0.082 point sur LightGBM classique, ce qui
valide l'hypothese locale lineaire. Elles restent toutefois legerement sous la
V2 et ne franchissent pas le gate.

A la demande de validation supplementaire, une exception au gate a ete faite:
le LightGBM lineaire compact a ete execute sur les huit folds. Chaque modele est
entraine sur environ sept huitiemes des dates, soit environ 461 000 lignes.

| Modele complet | Accuracy | Score penalise TS |
|---|---:|---:|
| LightGBM lineaire compact, 8 folds | 0.524442 | 0.522852 |
| Logistique V2, 8 folds | **0.525001** | **0.523285** |

Le screening sur deux folds etait donc legerement optimiste. Le resultat complet
confirme le rejet du modele lineaire.

Avec 32 variables returns, comprenant davantage de statistiques transversales,
le modele lineaire descend a 0.523737 contre 0.524258 pour les feuilles
constantes. Les feuilles lineaires sont donc sensibles a l'ajout de variables
bruitees.

### Blocs larges

| Bloc | Feuilles constantes | Feuilles lineaires |
|---|---:|---:|
| Combined, 80 variables | 0.523994 | 0.522316 |
| Non-return initial contamine par `RET_1 x turnover` | 0.524372 | 0.521674 |

La definition du bloc non-return a ensuite ete corrigee pour supprimer toute
variable utilisant un rendement.

## Signal hors rendements et categories pertinentes

Le bloc non-return propre contient volumes signes, turnover, missingness et
statistiques par date/groupe.

| Categories disponibles | Accuracy 2 folds |
|---|---:|
| `GROUP` | 0.512252 |
| `GROUP` + cluster allocation X-only | 0.514708 |
| `ALLOCATION` + `GROUP` + cluster | 0.515682 |

Le cluster non supervise gagne 0.246 point face a `GROUP` seul. Il est construit
uniquement avec les profils de volumes, turnover et missingness observes dans
le train du fold. Il extrait donc bien des archetypes pertinents, mais leur
signal autonome reste trop faible pour concurrencer les returns.

Les variables non-return les plus utilisees par LightGBM sont:

- dispersion du turnover par date;
- dispersion de la moyenne des volumes par date;
- moyennes et dispersions de `SIGNED_VOLUME_1` et `SIGNED_VOLUME_2`;
- statistiques correspondantes au niveau `TS x GROUP`.

## Quantile Regression Forest

Le smoke test utilise un fold, 20 arbres, profondeur 8 et feuilles d'au moins
500 observations. Il prend environ 85 secondes.

| Strategie | Accuracy fold 1 |
|---|---:|
| Moyenne conditionnelle, seuil zero | 0.513228 |
| Mediane conditionnelle, seuil zero | 0.510228 |
| Logistique V2 | 0.525077 |

L'intervalle conditionnel 40-60 % ne se trouve entierement d'un cote de zero
que pour 0.43 % des observations. L'accuracy sur ces observations dites
confiantes est 0.486. Les intervalles 25-75 % et 10-90 % ne declarent aucune
observation confiante. Le fallback reproduit donc la logistique sans gain.

## Gates et decision

- aucun LightGBM ne bat la V2; le meilleur a ete verifie sur huit folds;
- QRF est largement sous la V2 et trop couteux;
- l'exception LightGBM a huit folds confirme le gate initial;
- Cubist et Local Linear Forest ne sont pas declenches, conformement au gate;
- aucune soumission V3 inferieure n'est produite.

La V2 logistique reste le meilleur modele propre et robuste de cette recherche.

## Reproductibilite

- `gpt/v3_features.py`: blocs de features et clusters X-only;
- `gpt/lightgbm_v3.py`: comparaison constant/linear tree;
- `gpt/qrf_v3.py`: moyenne, quantiles et fallback;
- `gpt/requirements-v3.txt`: dependances isolees;
- sorties de screening dans `gpt/outputs/v3_*`.
