# H12 — résultats des modèles puissants

## Factorization machine low-rank

Un correcteur bilinéaire de rang 4 a été ajusté au-dessus du logit V2 exact :

\[
z_{i,t}=z^{V2}_{i,t}+w^\top m_t+u_i^\top V m_t.
\]

Sur les deux folds de screening :

- V2 : 0,525112 ;
- low-rank : 0,519385 ;
- perte : -0,005727 ;
- IC 95 % apparié : [-0,009642 ; -0,001813] ;
- écart train–validation final : 0,0163.

L'overfit apparaît dès le premier epoch et s'amplifie ensuite. Le gate bloque
les huit folds, les rangs 2/8 et toute architecture plus profonde.

## LightGBM conditionnel fortement régularisé

Le modèle utilise huit returns, huit états, `ALLOCATION` et `GROUP`. La
configuration impose profondeur 3, au plus 8 feuilles, 5 000 observations
minimum par feuille, L1/L2, path smoothing, régularisation catégorielle,
subsampling et feature sampling.

| Feuilles | Accuracy 8 folds | AUC | Gap train–validation | Profondeur max |
|---|---:|---:|---:|---:|
| constantes | 0,523723 | 0,534745 | 0,0108 | 3 |
| linéaires | 0,524072 | 0,534885 | 0,0115 | 3 |
| V2 | **0,525001** | ~0,5354 | ~0,0045 | — |

Le LightGBM linéaire avait légèrement gagné les deux premiers folds, puis perd
sur l'ensemble. La profondeur réelle confirme que le diagnostic n'est pas
« les arbres sont accidentellement trop profonds ». Malgré une structure très
contrainte, ils exploitent davantage le train que V2 sans mieux généraliser.

## Décision

Aucun modèle H12 ne passe son gate. L'ensemble/stacking est volontairement
bloqué : mélanger un modèle plus faible uniquement après observation de ses OOF
ajouterait une nouvelle sélection opportuniste.

La leçon n'est pas que les modèles puissants sont inutiles. Elle est que notre
signal conditionnel est trop faible pour payer la variance d'embeddings libres
ou d'arbres, même fortement régularisés. La prochaine hypothèse doit apporter
une information ou une invariance nouvelle, pas seulement davantage de capacité.
