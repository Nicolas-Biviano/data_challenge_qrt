# H10 — résultats des rendements dé-factorisés

## Résultat principal

La baseline V2 est reproduite exactement à **0,525001**. Aucune neutralisation
ne l'améliore.

| Variante | Accuracy | Gain vs V2 | IC 95 % du gain par date | Folds gagnés |
|---|---:|---:|---:|---:|
| V2 brute | 0,525001 | 0 | — | — |
| Résidu au marché seul | 0,522869 | -0,002133 | [-0,003439 ; -0,000826] | 0/8 |
| Résidu au groupe seul | 0,521486 | -0,003516 | [-0,005006 ; -0,002026] | 0/8 |
| Brut + deux résidus | 0,523358 | -0,001643 | [-0,002873 ; -0,000413] | 0/8 |
| Composantes hiérarchiques | 0,523666 | -0,001336 | [-0,002724 ; 0,000052] | 2/8 |
| Composantes + pente locale brute | 0,523220 | -0,001782 | [-0,003083 ; -0,000480] | 0/8 |

Les intervalles sont calculés sur la différence de correction ligne à ligne,
puis agrégés par date. Ils mesurent donc directement l'incertitude sur le gain,
et non deux barres d'erreur marginales difficiles à comparer.

## Ce que la décomposition révèle

Pour les huit lags, l'écart-type du facteur de marché vaut environ 25,5 % de
celui du rendement brut. L'écart `GROUP - marché` vaut 32 à 35 %, et le résidu
propre conserve 92 à 93 % de l'écart-type brut. Les composantes communes ne
dominent donc pas la variance, mais leur suppression retire du signal utile :

- retirer seulement le marché coûte 0,213 point d'accuracy ;
- retirer marché et groupe coûte 0,352 point ;
- l'effet est défavorable dans chacun des huit folds.

On ne peut pas conclure que le marché ou `GROUP` est causal. La conclusion
précise est que, dans cette représentation linéaire et cette CV, la direction
future n'est pas mieux prédite à partir des seuls mouvements relatifs.

## Pourquoi l'ajout des résidus peut perdre malgré l'information brute

`raw_plus_defactored` contient mathématiquement beaucoup d'information
redondante. Après standardisation, la pénalisation L2 agit sur plusieurs
colonnes très corrélées et change le prior implicite sur les coefficients. Le
modèle peut donc obtenir une frontière différente même si le brut est encore
présent. Cette perte ne prouve pas que les résidus n'ont aucune information ;
elle montre qu'ils n'apportent pas assez d'information nouvelle pour compenser
la variance et la reparamétrisation.

Le contrôle `hierarchical_raw_local_slope` conserve la pente locale la plus
importante de V2 et perd tout de même les huit folds. La conclusion ne vient
donc pas seulement du remplacement de `ALLOCATION × RET_1`.

## Décision

Le gate H10 échoue. On ne lance ni PCA factorielle, ni soumission, ni tuning de
la régularisation sur cette famille. Les features peuvent rester utiles pour
des diagnostics de régime, mais elles ne remplacent pas les returns bruts.

La prochaine piste devrait conserver explicitement les composantes communes et
chercher un signal nouveau, plutôt que neutraliser davantage les returns.
