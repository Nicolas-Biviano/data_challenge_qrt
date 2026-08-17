# 09 - `SIGNED_VOLUME_1` avec un modele deux-parties

## Motivation

`SIGNED_VOLUME_1` semble utile mais manque sur 73.52 % du train. Une imputation
standard melange alors deux phenomenes differents :

1. le volume est-il disponible ?
2. quelle est sa valeur lorsqu'il est disponible ?

Cette experience separe explicitement les deux parties. Elle n'utilise ni
target encoding, ni ordre des dates, ni reconstruction des fenetres.

## Audit initial

| Sous-population | Lignes | Taux de cible positive | Moyenne de la cible |
|---|---:|---:|---:|
| `SIGNED_VOLUME_1` manquant | 387 506 | 0.503337 | 0.000003 |
| `SIGNED_VOLUME_1` observe | 139 567 | 0.517866 | 0.000123 |

La disponibilite identifie donc un regime different. Cependant, parmi les
lignes observees :

- la correlation entre la valeur brute et la cible continue vaut seulement
  0.00428 ;
- `sign(SIGNED_VOLUME_1)` predit le signe de la cible avec 0.50348 d'accuracy ;
- la relation par decile n'est pas monotone.

La V2 confirme ce diagnostic. Elle atteint 0.531257 sur les lignes observees,
mais sa probabilite moyenne y sous-estime le taux positif de 0.00930. La
disponibilite contient donc surtout du signal de regime ; la valeur brute ne
fournit pas une bonne pente directionnelle globale.

## Modele global deux-parties

Le script `gpt/volume_two_part.py` teste :

- indicateur de presence de `SIGNED_VOLUME_1` ;
- parts de volumes disponibles sur 5 et 20 lags ;
- taux de disponibilite dans `TS` et dans `TS x GROUP` ;
- valeur signee compressee avec `sign(x) log(1 + abs(x))` ;
- valeur absolue compressee et signe ;
- rangs au sein de `TS` et `TS x GROUP` ;
- interactions avec `RET_1` ;
- interactions de presence avec `GROUP` et `ALLOCATION`.

Toutes les transformations apprises par le modele sont reajustees dans le
fold. Les rangs et taux sont calcules avec `X` seulement.

Resultats sur les deux folds de screening, avec `C=0.003` :

| Variante | Accuracy | Score penalise TS |
|---|---:|---:|
| V2 reproduite | **0.525112** | 0.521739 |
| Deux-parties, rangs | 0.525059 | **0.521744** |
| Presence seule | 0.525029 | 0.521711 |
| Presence x GROUP | 0.525006 | 0.521657 |
| Deux-parties complet | 0.524832 | 0.521488 |
| Valeur compressee | 0.524621 | 0.521317 |

Le meilleur score brut reste la V2. La valeur de volume et les interactions
complexes ajoutent surtout du bruit.

## Specialiste conditionnel

Le script `gpt/volume_specialist.py` ajuste deux modeles dans chaque fold :

- la V2 sur tout le train, utilisee comme fallback lorsque le volume manque ;
- un expert logistique ajuste uniquement sur les lignes ou le volume est
  observe, utilise uniquement dans ce meme regime en validation.

Le meilleur expert conserve seulement les huit rendements de la V2. Ajouter la
valeur ou les rangs du volume degrade le score. C'est donc la selection d'un
regime par la disponibilite, et non la valeur du volume, qui apporte le signal.

### Screening deux folds

| Expert | C | Accuracy | Gain vs V2 |
|---|---:|---:|---:|
| Rendements seuls | 0.003 | **0.525988** | +0.000876 |
| Rendements + rangs | 0.003 | 0.525800 | +0.000688 |
| Complet | 0.003 | 0.525520 | +0.000408 |

### Validation huit folds du seul meilleur expert

| Mesure | V2 | Specialiste route | Difference |
|---|---:|---:|---:|
| Accuracy globale | 0.525001 | **0.525185** | +0.000184 |
| Score penalise TS | 0.523285 | **0.523483** | +0.000198 |
| Erreur standard allocation | 0.001353 | **0.001345** | -0.000008 |

Le specialiste ameliore l'accuracy des lignes observees de 0.531257 a 0.531952
(+0.000695). Les lignes manquantes sont strictement identiques a la V2.

Les gains par fold restent instables :

| Fold | Difference d'accuracy vs V2 |
|---:|---:|
| 1 | +0.002470 |
| 2 | -0.000708 |
| 3 | +0.002216 |
| 4 | -0.002494 |
| 5 | +0.001869 |
| 6 | -0.001102 |
| 7 | -0.000634 |
| 8 | -0.000199 |

Seulement quatre folds sur huit progressent. Un intervalle apparié par date du
gain contient largement zero. Le gain global est favorise par la taille des
dates sur lesquelles le specialiste fonctionne mieux ; il ne constitue pas une
amelioration statistiquement solide.

## Test de shrinkage hors fold

Un poids entre la probabilite V2 et celle du specialiste a ete choisi pour
chaque fold sur les sept autres folds, parmi 0.0, 0.1, ..., 1.0. Cette procedure
hors fold atteint seulement 0.524574, soit -0.000427 sous V2. Les poids choisis
varient de 0.1 a 1.0, confirmant l'instabilite.

## Decision

La disponibilite de `SIGNED_VOLUME_1` identifie bien un regime distinct, mais :

- la valeur observee elle-meme n'ajoute pas de signal directionnel robuste ;
- le petit gain du specialiste n'est present que sur quatre folds ;
- un shrinkage hors fold ne le stabilise pas.

La V2 reste le modele principal. Le specialiste est conserve comme candidat de
diversification future, mais ne doit pas remplacer seul la soumission V2.

## Artefacts

- `gpt/volume_two_part.py` ;
- `gpt/volume_specialist.py` ;
- `gpt/outputs/volume_two_part_screen/` ;
- `gpt/outputs/volume_specialist_screen/` ;
- `gpt/outputs/volume_specialist_full8/`.
