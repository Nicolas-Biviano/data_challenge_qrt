# H14 — Résultats du screening de cible transversale

## Verdict

Le gate échoue nettement. Sur quatre folds de dates, la cible brute atteint
0,525981. La meilleure cible transformée en accuracy, `market_100`, atteint
0,525103 : **−0,000878**, soit −0,088 point. Elle perd les quatre folds.

| Cible | Accuracy | AUC | Gain d'accuracy | Folds gagnés |
|---|---:|---:|---:|---:|
| raw | 0,525981 | 0,536595 | — | — |
| market_100 | 0,525103 | 0,536912 | −0,000878 | 0/4 |
| market_50 | 0,524973 | 0,536784 | −0,001008 | 0/4 |
| hierarchical_50 | 0,524816 | **0,537035** | −0,001165 | 0/4 |
| date_zscore_clip3 | 0,524518 | 0,536801 | −0,001462 | 1/4 |
| date_zscore | 0,524503 | 0,536672 | −0,001478 | 1/4 |
| hierarchical_100 | 0,524339 | 0,536789 | −0,001642 | 0/4 |

L'IC95 apparié du gain de `market_100` vaut
`[-0,002167 ; +0,000411]`. Le modèle passe de 54,41 % à 49,17 % de prédictions
positives.

## Interprétation

Retirer le facteur date ou le tilt de groupe améliore légèrement l'AUC : la
meilleure AUC est celle de `hierarchical_50`, à +0,000440 au-dessus de la cible
brute. En revanche, toutes les variantes perdent en accuracy à seuil zéro.

La composante commune retirée semble donc partiellement imprévisible mais reste
utile pour localiser le niveau absolu du rendement futur. La dé-factorisation
transforme mieux le classement relatif que le signe absolu demandé par le
challenge. Le recentrage du taux positif n'est pas, à lui seul, une amélioration.

## Décision

- ne pas lancer les huit folds ;
- ne pas construire de soumission ;
- ne pas apprendre un seuil après observation de ces folds, ce qui recyclerait
  la validation ;
- conserver l'idée uniquement si une future tâche vise le ranking transversal
  ou une stratégie long-short neutre, pas l'accuracy du signe brut.
