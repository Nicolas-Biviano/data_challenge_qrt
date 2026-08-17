# Formulation mathematique du modele GPT V2

## Variables

Pour une observation associee a la date anonymisee `t` et a l'allocation `a`,
on utilise le vecteur de rendements selectionnes

$$
x_{a,t} = (RET_1, RET_2, RET_3, RET_4, RET_7, RET_8, RET_9, RET_{18})_{a,t}.
$$

Les composantes continues sont imputees par leur mediane d'entrainement puis
standardisees dans chaque fold. On note le resultat $z_{a,t}$.

## Score lineaire

Le score logistique est

$$
\eta_{a,t}
= \beta_0
+ z_{a,t}^{\top}\beta
+ \alpha_a
+ \gamma_{g(a)}
+ \delta_a z_{a,t,RET_1},
$$

avec:

- $\alpha_a$: effet fixe de l'allocation;
- $\gamma_{g(a)}$: effet fixe du groupe;
- $\delta_a$: sensibilite propre de l'allocation a `RET_1`.

La probabilite predite d'un rendement futur positif vaut

$$
\hat p_{a,t}=\sigma(\eta_{a,t})
=\frac{1}{1+\exp(-\eta_{a,t})}.
$$

La classe finale est

$$
\hat y_{a,t}=\mathbf{1}[\hat p_{a,t}>0.5].
$$

## Estimation

Les coefficients sont obtenus par minimisation de la log-vraisemblance
penalisee:

$$
\min_{\theta}
-\sum_i\left[y_i\log(\hat p_i)+(1-y_i)\log(1-\hat p_i)\right]
+\lambda\lVert\theta\rVert_2^2,
$$

ou la force de regularisation est controlee par $C=0.003$ dans scikit-learn
($C$ est inversement lie a la regularisation).

## Validation

Les valeurs uniques de `TS` sont reparties aleatoirement dans huit folds. Une
date complete appartient a un seul fold. Aucun ordre numerique des labels
`DATE_xxxx` n'est utilise et aucun target encoding n'est construit.

