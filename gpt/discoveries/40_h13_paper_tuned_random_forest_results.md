# H13 — Résultats de la Random Forest très régularisée

## Verdict

La Random Forest ne remplace pas V2 : elle atteint **0,523711** d'accuracy OOF
contre **0,525001** pour V2. Le gain apparié est de **−0,001290**, soit
**−0,129 point de pourcentage**.

L'erreur standard du gain calculée entre dates est de 0,001097 et son intervalle
à 95 % est `[-0,003441 ; +0,000860]`. La forêt gagne quatre folds sur huit :
l'expérience n'établit ni un gain, ni même une dégradation certaine, mais son
estimation centrale est nettement défavorable.

| Mesure OOF | Random Forest | V2 |
|---|---:|---:|
| Accuracy | 0,523711 | 0,525001 |
| AUC | 0,532849 | 0,535216 |
| Brier | 0,249118 | 0,248932 |
| Taux de prédictions positives | 58,47 % | 60,60 % |

Le taux cible OOF est 50,72 %. La forêt réduit le biais positif de V2 sans le
résoudre, mais perd également en classement (AUC) et en calibration (Brier).

## Pourquoi ce tuning est « propre »

Le protocole suit quatre idées de la littérature primaire :

1. Probst, Wright et Boulesteix identifient `mtry`/`max_features`, la fraction
   d'échantillonnage et la taille des nœuds parmi les principaux contrôles du
   compromis force–corrélation. Ces paramètres sont tunés ensemble, à
   l'intérieur de chaque fold externe.
2. Mentch et Zhou interprètent la randomisation des variables comme une forme
   de régularisation, particulièrement pertinente à faible rapport signal/bruit.
3. Le nombre d'arbres n'est pas choisi au meilleur point observé : il est fixé à
   1 000, conformément à la recommandation de Probst et Boulesteix de contrôler
   la convergence plutôt que de tuner opportunément ce nombre.
4. Le score OOB de Breiman est enregistré comme diagnostic interne, mais jamais
   utilisé à la place de la validation par dates.

La sélection est strictement imbriquée. Chaque fold externe contient un
train/validation interne par dates pour choisir une configuration ; le fold
externe ne sert qu'une fois à l'estimation finale. Aucun target encoding n'est
utilisé.

## Régularisation réellement obtenue

Cinq folds choisissent `pruned` (profondeur maximale 6, au plus 24 feuilles,
0,25 % d'observations par feuille, `max_features=0,35`, bootstrap 50 %, seuil
d'impureté et `ccp_alpha`). Trois folds choisissent `ultra_random` (profondeur
3, au plus 8 feuilles, feuilles de 2 %, `max_features=0,25`, bootstrap 40 %).

- `pruned` atteint réellement une profondeur 6 et environ 23,8 feuilles ; le
  quantile 5 % des tailles de feuille est proche de 1 188 observations.
- `ultra_random` atteint une profondeur 3 et seulement 6,2 feuilles en moyenne ;
  le quantile 5 % des feuilles dépasse 9 280 observations.
- l'écart train–validation externe moyen n'est que de **0,415 point**.

On n'observe donc pas le surapprentissage violent des LightGBM précédents. Le
problème est plutôt un compromis biais–variance qui change selon les dates : la
forêt perd fortement sur les folds 3, 4 et 6, puis regagne sur 7 et 8.

| Fold | Configuration | RF | V2 | Gain RF − V2 |
|---:|---|---:|---:|---:|
| 1 | pruned | 0,525637 | 0,525077 | +0,000561 |
| 2 | pruned | 0,527272 | 0,525147 | +0,002125 |
| 3 | pruned | 0,522271 | 0,527720 | −0,005449 |
| 4 | ultra_random | 0,523937 | 0,530334 | −0,006397 |
| 5 | ultra_random | 0,518817 | 0,519648 | −0,000831 |
| 6 | ultra_random | 0,523562 | 0,529708 | −0,006145 |
| 7 | pruned | 0,526152 | 0,521832 | +0,004320 |
| 8 | pruned | 0,522085 | 0,520769 | +0,001316 |

## Nombre d'arbres

Les moyennes externes passent de 0,523426 à 100 arbres à 0,523589 à 300,
0,523712 à 600 puis 0,523717 à 1 000. L'AUC passe de 0,532954 à 0,533374.
La courbe est pratiquement stabilisée vers 600 arbres : davantage d'arbres
réduirait surtout le bruit Monte-Carlo, sans raison d'espérer combler 0,129
point d'accuracy.

## Ce que racontent les variables

L'importance par diminution d'impureté place en tête `RET_1`, le profil moyen
de `RET_1` de l'allocation, les deux résidus de `RET_1`, puis les états de marché
et de volatilité. Cela confirme que la forêt utilise réellement l'information
de style et les relations transversales, mais sans améliorer V2.

Cette importance est biaisée en faveur des variables continues et corrélées.
Elle est donc utilisée uniquement pour comprendre le fit, pas pour déclarer une
feature « vraie », ni pour lancer une sélection sur les mêmes OOF.

## Conclusion opérationnelle

- ne pas soumettre cette forêt seule ;
- ne pas l'empiler automatiquement : elle échoue au gate de performance ;
- conserver ses probabilités OOF pour une future analyse de complémentarité
  conditionnelle, seulement si un gate indépendant justifie un ensemble ;
- la prochaine expérience doit chercher une représentation plus invariante des
  relations allocation–marché, pas simplement une forêt plus profonde.

## Sources

- Probst, Wright et Boulesteix, [Hyperparameters and Tuning Strategies for
  Random Forest](https://arxiv.org/abs/1804.03515).
- Mentch et Zhou, [Randomization as
  Regularization](https://www.jmlr.org/papers/v21/19-905.html).
- Probst et Boulesteix, [To tune or not to tune the number of
  trees](https://arxiv.org/abs/1705.05654).
- Breiman, [Random Forests](https://doi.org/10.1023/A:1010933404324).
