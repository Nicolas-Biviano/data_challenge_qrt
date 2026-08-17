# Phase modèles puissants — protocole

## Principe

Un modèle plus puissant n'est utile que s'il représente mieux une hypothèse
économique précise. La phase suivante cible la réponse conditionnelle propre aux
allocations mise en évidence faiblement par H11.

## Étape A — robustesse de H11

- CV par dates, 8 folds ;
- seeds de découpage `0, 1, 2, 3, 4` ;
- V2 et H11 gelés, comparés sur les mêmes folds ;
- aucun choix du meilleur seed ;
- gain apparié moyen par date après moyenne des cinq répétitions.

H11 est jugé robuste si son gain moyen est positif sur au moins quatre seeds,
si au moins 60 % des 40 folds sont gagnés et si le gain moyen ne repose pas sur
un seul découpage.

## Étape B — factorization machine low-rank

Le modèle généralise V2 :

\[
z_{i,t}=a_i+\beta^\top x_{i,t}+b_i RET_{1,i,t}
+u_i^\top V m_t.
\]

`u_i` est un embedding de faible dimension de l'allocation et `V m_t` une
projection de l'état observable. Le produit bilinéaire apprend une matrice de
réponses `allocation × état` de rang faible. Aucun target encoding n'est
utilisé : embeddings et coefficients sont des paramètres régularisés du modèle.

Gates : comparaison à V2 et H11, écart train-validation, sensibilité au rang et
à l'initialisation. Le rang reste petit (`2, 4, 8`) et aucune architecture
profonde n'est autorisée avant un gain stable.

## Étape C — LightGBM conditionnel

Un LightGBM fortement régularisé reçoit returns bruts, états et catégories. Le
diagnostic doit reporter profondeur réelle, feuilles, train-validation, AUC et
accuracy. Les configurations peu profondes sont prioritaires ; les paramètres
par défaut ne constituent pas une baseline acceptable.

## Étape D — ensemble

Un ensemble n'est testé que si un nouveau modèle :

- gagne au moins cinq folds ;
- apporte des erreurs suffisamment différentes de V2 ;
- ne dégrade pas les panels de 120 dates test-like.

Le stacking utilise exclusivement des prédictions OOF. Aucun modèle de niveau 2
n'est ajusté sur des prédictions in-sample.
