# H11 — fonctions de réaction conditionnelles des allocations

## Reformulation de l'hypothèse

H8–H10 ne montrent pas que les régimes sont inutiles. Ils montrent que nous ne
savons pas prévoir une étiquette globale de régime ni améliorer V2 en retirant
les composantes communes.

H11 teste une autre question : connaissant l'état observable dans les lags,
peut-on apprendre que chaque `GROUP` ou `ALLOCATION` y réagit différemment ?

\[
\operatorname{logit}P(y_{i,t+1}>0)
=a_i+\beta^\top x_{i,t}
+\gamma_{g(i)}^\top m_t
+\delta_i^\top m_t.
\]

`m_t` n'est pas le régime futur. C'est un petit vecteur d'état construit avec
les returns retardés déjà disponibles à la date de prédiction.

## États X-only

Chaque agrégat exclut la ligne courante :

- moyenne transversale de `RET_1` ;
- dispersion transversale de `RET_1` ;
- breadth : proportion d'allocations avec `RET_1 > 0` ;
- momentum transversal court : moyenne `RET_1:3` moins moyenne `RET_4:10` ;
- volatilité récente moyenne des allocations (`RET_1:5`) ;
- niveau `RET_1` du groupe relativement au marché ;
- momentum court du groupe relativement au marché ;
- dispersion `RET_1` du groupe relativement au marché.

Les dates contiennent environ 209 lignes et les cellules `date × GROUP` environ
69 lignes. Aucun target encoding ni ordre chronologique n'est utilisé.

## Modèles

Tous conservent V2 : huit returns bruts, effets fixes `ALLOCATION` et `GROUP`,
et pente locale `ALLOCATION × RET_1`.

1. `baseline_raw` : V2 exacte ;
2. `state_main` : ajoute seulement les huit états ;
3. `group_response` : ajoute `GROUP × état` ;
4. `allocation_response_strong` : ajoute `ALLOCATION × état`, avec échelle 0,20 ;
5. `hierarchical_response` : effets globaux + groupe (échelle 0,50) + allocation
   (échelle 0,20).

Après le premier passage, le seul bloc positif (`ALLOCATION × état`, échelle
0,20) autorise un diagnostic borné de shrinkage à 0,10 et 0,35. Cette courbe
est exploratoire : son meilleur point ne constitue pas une estimation non
biaisée d'un modèle sélectionné.

Avec une pénalisation L2, multiplier un bloc par 0,20 impose approximativement
25 fois plus de pénalisation sur son effet prédictif. Cela permet aux réponses
par allocation de ne s'écarter de la réponse commune que si les données le
justifient.

## Diagnostic obligatoire

- accuracy et AUC OOF ;
- écart train–validation pour détecter l'overfit ;
- gain apparié par date, erreur standard et IC 95 % ;
- nombre de folds gagnés ;
- taux de prédictions positives.

Le test est arrêté sans recherche d'états supplémentaires si aucune interaction
ne gagne au moins cinq folds ou si tous les gains appariés sont clairement
défavorables. Si un bloc passe, la prochaine étape sera une factorisation
low-rank des réponses `ALLOCATION × état`, pas une multiplication manuelle de
centaines de transformations.
