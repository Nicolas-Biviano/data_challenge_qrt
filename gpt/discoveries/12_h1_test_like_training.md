# H1 — entraînement sur les dates test-like

## Hypothèse

Le test est dominé par des dates de 278 lignes. Les dates train de 276 lignes
sont le régime observable le plus proche. On teste si entraîner uniquement sur
ces dates améliore la généralisation sur des dates complètes tenues à l'écart.

## Protocole

- mêmes folds de validation pour toutes les variantes ;
- dates entières dans les folds ;
- comparaison appariée par date ;
- bootstrap des dates pour les intervalles à 95 % ;
- aucun ordre des dates et aucun target encoding.

## Résultat

La référence entraînée sur toutes les dates atteint 0,518941. La meilleure
variante entraînée uniquement sur les dates complètes, avec `C=0.01`, atteint
0,518318. Sa différence est de -0,000623, avec un intervalle à 95 % de
[-0,002801 ; 0,001566].

## Décision

H1 n'est pas soutenue. La taille de date révèle un décalage de distribution,
mais filtrer le train n'améliore pas la relation entre X et la cible. Le modèle
final doit continuer à exploiter toutes les dates.

Notebook : `gpt/notebooks/H1_test_like_training.ipynb`.
