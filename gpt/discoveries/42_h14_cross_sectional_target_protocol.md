# H14 — Cible continue transversalement normalisée

## Hypothèse

Les meilleurs modèles publics connus apprennent la cible continue avant de
prendre son signe. La composante commune à une date peut toutefois être plus
difficile à prévoir que la position relative d'une allocation. H14 teste donc
si retirer tout ou partie de cette composante **dans la cible d'entraînement**
rend la relation plus transférable.

Ce test est différent de H10 : H10 dé-factorisait les returns passés utilisés
comme features ; H14 dé-factorise le rendement futur utilisé comme label de
régression.

## Cibles testées

Pour une ligne `i`, une date `t` et un groupe `g`, on note :

- `y_it` la cible continue brute ;
- `m_t` sa moyenne dans la date d'entraînement ;
- `m_gt` sa moyenne dans le groupe et la date ;
- `s_t` son écart-type dans la date.

Les sept variantes pré-enregistrées sont :

| Nom | Cible ajustée |
|---|---|
| `raw` | `y_it` |
| `market_50` | `y_it - 0.5 m_t` |
| `market_100` | `y_it - m_t` |
| `hierarchical_50` | `y_it - m_t - 0.5 (m_gt - m_t)` |
| `hierarchical_100` | `y_it - m_gt` |
| `date_zscore` | `(y_it - m_t) / s_t` |
| `date_zscore_clip3` | `clip((y_it - m_t) / s_t, -3, 3)` |

Les transformations de cible sont calculées uniquement dans le train du fold.
À la validation et au test, le modèle produit directement un score résiduel et
son signe est comparé au signe de la cible brute. Aucune statistique de la cible
future n'est requise.

## Modèle contrôlé

La matrice est exactement celle de V1 :

- huit returns ;
- effets fixes `GROUP` et `ALLOCATION` ;
- pente `ALLOCATION × RET_1` ;
- Ridge `alpha=100`.

Ainsi, un éventuel gain ne peut venir que de la définition de la cible.

## Validation et gate

1. Screening sur quatre folds de dates, seed 0.
2. Comparaison appariée à `raw` sur exactement les mêmes lignes.
3. Passage aux huit folds si le meilleur challenger :
   - gagne au moins 0,0002 d'accuracy ;
   - gagne au moins 3 folds sur 4 ;
   - ne perd pas plus de 0,0002 d'AUC.
4. Si les huit folds passent, répétition sur plusieurs seeds dans H16.

Les gains seront accompagnés de leur erreur standard appariée par date. Un
screening positif n'est jamais présenté comme une conclusion.
