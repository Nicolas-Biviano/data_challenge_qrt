# Version mathematique du modele

Ce document formalise la tentative GPT implementee dans `gpt/model.py`.

## Notations

Pour chaque observation \(i\), on observe :

- une date anonymisee \(t_i = TS_i\) ;
- une allocation \(a_i = ALLOCATION_i\) ;
- un groupe \(g_i = GROUP_i\) ;
- des rendements retardes \(r_{i,k} = RET_k\), avec
  \(k \in \{1,2,3,4,7,8,9,18\}\) ;
- une cible continue \(y_i\) ;
- une cible binaire

\[
z_i = \mathbf{1}\{y_i > 0\}.
\]

Le modele est entraine sur la cible continue \(y_i\), puis transforme en
prediction binaire par seuillage du score predit.

## Standardisation numerique

Dans chaque fold de cross-validation, les variables numeriques sont imputees
par la mediane du train puis standardisees :

\[
\tilde r_{i,k}
= \frac{r_{i,k} - \mu_k}{\sigma_k},
\]

ou \(\mu_k\) et \(\sigma_k\) sont estimes uniquement sur le train du fold.

Cette etape est importante : le preprocessing est refit dans chaque fold, donc
la validation reste out-of-fold.

## Matrice de design

La prediction continue est construite a partir de trois blocs.

### 1. Effets numeriques communs

\[
x_i^{num}
=
\left(
\tilde r_{i,1},
\tilde r_{i,2},
\tilde r_{i,3},
\tilde r_{i,4},
\tilde r_{i,7},
\tilde r_{i,8},
\tilde r_{i,9},
\tilde r_{i,18}
\right).
\]

### 2. Effets fixes d'allocation et de groupe

On encode l'allocation et le groupe en one-hot :

\[
e(a_i) \in \{0,1\}^{|\mathcal A|},
\qquad
e(g_i) \in \{0,1\}^{|\mathcal G|}.
\]

Ces termes autorisent un intercept different selon l'allocation et le groupe.

### 3. Interaction allocation x RET_1

Le modele ajoute une pente specifique par allocation pour le signal court terme
\(RET_1\) :

\[
e(a_i)\tilde r_{i,1}
=
\left(
\mathbf{1}\{a_i=a\}\tilde r_{i,1}
\right)_{a \in \mathcal A}.
\]

Cela permet au modele d'apprendre que \(RET_1\) peut etre plus ou moins
predictif selon l'allocation.

## Score continu predit

Le score predit s'ecrit :

\[
\hat y_i
=
\beta_0
+ \sum_{k \in K} \beta_k \tilde r_{i,k}
+ \alpha_{a_i}
+ \gamma_{g_i}
+ \delta_{a_i}\tilde r_{i,1},
\]

avec :

\[
K = \{1,2,3,4,7,8,9,18\}.
\]

Interpretation des coefficients :

- \(\beta_k\) : effet commun du rendement retarde \(RET_k\) ;
- \(\alpha_{a_i}\) : effet fixe de l'allocation ;
- \(\gamma_{g_i}\) : effet fixe du groupe ;
- \(\delta_{a_i}\) : correction locale de la pente de \(RET_1\) pour
  l'allocation \(a_i\).

En notation matricielle :

\[
\hat y = X\theta.
\]

## Estimation Ridge

Les coefficients sont estimes par regression Ridge :

\[
\hat\theta
=
\arg\min_{\theta}
\left[
\sum_{i \in \mathcal T}
\left(y_i - x_i^\top \theta\right)^2
+ \lambda \|\theta\|_2^2
\right],
\]

avec :

\[
\lambda = 100.
\]

Dans le code, le solveur utilise :

```python
Ridge(alpha=100.0, solver="lsqr")
```

La regularisation shrinke les effets fixes et les pentes specifiques vers des
valeurs plus moderees, ce qui limite le sur-apprentissage sur les allocations
moins documentees.

## Prediction binaire

La prediction finale pour le challenge est obtenue par le signe du score
continu :

\[
\hat z_i
=
\mathbf{1}\{\hat y_i > 0\}.
\]

Donc :

- si \(\hat y_i > 0\), on predit `1` ;
- sinon, on predit `0`.

## Cross-validation

Le framework `src.cross_validation.run_cv` fait une validation croisee par
dates entieres. Les dates \(TS\) sont reparties entre les folds, puis toutes
les observations d'une meme date restent ensemble.

Pour chaque observation de validation, on obtient une prediction out-of-fold :

\[
\hat z_i^{OOF}.
\]

L'accuracy globale est :

\[
Accuracy
=
\frac{1}{n}
\sum_{i=1}^{n}
\mathbf{1}\{\hat z_i^{OOF} = z_i\}.
\]

Le score penalise utilise dans les rapports est :

\[
Score_G
=
Accuracy
-
\frac{\operatorname{std}_{g \in G}(Accuracy_g)}{\sqrt{|G|}},
\]

ou \(G\) peut etre :

- les folds ;
- les dates \(TS\) ;
- les allocations.

Cela penalise un modele dont la performance est trop instable selon l'axe
d'analyse choisi.

## Baseline comparee

La baseline naive utilise seulement le signe de \(RET_1\) :

\[
\hat z_i^{base}
=
\mathbf{1}\{RET_{i,1} > 0\}.
\]

La tentative Ridge conserve ce signal mais lui ajoute :

- des effets fixes d'allocation ;
- des effets fixes de groupe ;
- une pente allocation-specifique sur \(RET_1\) ;
- quelques rendements retardes supplementaires.

