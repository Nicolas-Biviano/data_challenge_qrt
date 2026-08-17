# H15 — Réponse hiérarchique d'allocation en très basse dimension

## Hypothèse

H11 suggère un minuscule signal dans les interactions allocation–état, mais ses
huit états produisent des centaines d'interactions corrélées. H15 réduit la
représentation à deux ou trois états pré-spécifiés et impose un shrinkage plus
fort aux déviations d'allocation qu'aux réponses de groupe.

## États retenus

- `market_ret1_mean` : direction du marché transversal ;
- `market_ret1_dispersion` : dispersion/régime de risque ;
- `group_ret1_relative` : tilt du groupe relativement au marché.

Ils sont construits leave-one-out à partir de `X` seulement. Aucun target
encoding ni ordre des dates n'est utilisé.

## Modèle hiérarchique

La V2 reste le socle. Pour chaque état `s`, on ajoute :

- une pente globale de `s` ;
- une déviation `GROUP × s`, fortement pénalisée ;
- une déviation `ALLOCATION × s`, encore plus pénalisée.

Dans la matrice standardisée, diminuer l'échelle d'un bloc augmente sa pénalité
L2 effective. Trois spécifications seulement sont pré-enregistrées :

| Nom | États | échelle groupe | échelle allocation |
|---|---:|---:|---:|
| `allocation_lowdim` | 3 | 0 | 0,20 |
| `hierarchical_ultra` | 3 | 0,15 | 0,05 |
| `hierarchical_directional` | 2 | 0,25 | 0,08 |
| `hierarchical_strong` | 3 | 0,30 | 0,10 |

Le premier isole l'effet de la réduction dimensionnelle. Les trois autres
testent un partial pooling `global → GROUP → ALLOCATION` de force croissante.

## Validation

Le screening utilise une CV 4-fold couvrant toutes les dates, seed 0. Le passage
à 8 folds exige :

- gain supérieur à 0,0002 ;
- au moins 3 folds gagnés sur 4 ;
- AUC non inférieure de plus de 0,0002 ;
- gap train–validation inférieur à 0,01.

Si le gate passe, la spécification est gelée avant huit folds puis H16 la répète
sur plusieurs seeds. Aucun hyperparamètre intermédiaire n'est choisi après
lecture des résultats.
