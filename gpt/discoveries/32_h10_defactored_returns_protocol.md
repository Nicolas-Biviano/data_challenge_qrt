# H10 — rendements dé-factorisés

## Hypothèse

Le rendement passé d'une allocation contient trois morceaux :

\[
RET_{i,t-h}=M_{t-h}+G_{g(i),t-h}+\varepsilon_{i,t-h},
\]

où `M` est le mouvement commun de la date, `G` l'écart du groupe au marché et
`epsilon` le mouvement propre à l'allocation. Le mouvement propre peut être plus
prévisible que le mouvement commun, ou apporter un signal complémentaire aux
rendements bruts.

Cette hypothèse concerne uniquement les `RET_*`. `SIGNED_VOLUME_*` n'est pas le
volume d'un titre et n'entre pas dans cette expérience.

## Construction X-only

Pour chacun des huit lags de la V2 et chaque ligne, les moyennes sont calculées
sans la ligne courante :

\[
\begin{aligned}
M^{(-i)}_{t,h} &= \operatorname{mean}_{j\ne i}(RET_{j,t-h}),\\
\bar R^{(-i)}_{g,t,h} &=
  \operatorname{mean}_{j\ne i,\,g(j)=g(i)}(RET_{j,t-h}),\\
G^{(-i)}_{g,t,h} &= \bar R^{(-i)}_{g,t,h}-M^{(-i)}_{t,h},\\
\varepsilon^{(-i)}_{i,t,h} &= RET_{i,t-h}-\bar R^{(-i)}_{g,t,h}.
\end{aligned}
\]

Les panels contiennent environ 209 allocations par date et 69 allocations par
`date × GROUP`; les moyennes sont donc suffisamment documentées. Le
leave-one-out évite que le retour propre d'une allocation ne soit recopié
mécaniquement dans son facteur.

Les agrégats utilisent seulement les features disponibles pour la date. Ils ne
consultent ni la target, ni les autres dates, ni un ordre temporel supposé. Les
dates de validation restent entièrement hors entraînement du modèle.

## Expériences atomiques

Toutes utilisent la logistique V2 (`C=0.003`), les mêmes effets fixes et les
mêmes huit folds de dates :

1. `baseline_raw` : V2 actuelle ;
2. `market_defactored_only` : résidus au marché à la place des returns bruts ;
3. `group_defactored_only` : résidus au groupe à la place des returns bruts ;
4. `raw_plus_defactored` : brut + résidu marché + résidu groupe ;
5. `hierarchical_components` : facteur marché + écart groupe + résidu propre.
6. `hierarchical_raw_local_slope` : même décomposition, mais conservation de
   la pente locale `ALLOCATION × RET_1 brut` comme contrôle d'ablation.

La pente locale `ALLOCATION × RET_1` porte sur la version de `RET_1` réellement
utilisée par chaque modèle. Dans la combinaison brut + résidus, elle reste sur
le `RET_1` brut pour conserver exactement la structure de la baseline.

## Critère de décision

On reporte obligatoirement : accuracy OOF, ROC-AUC, gain apparié par date,
erreur standard et intervalle à 95 %, ainsi que le nombre de folds gagnés.

Une variante ne passe à une soumission ou à une factorisation PCA que si :

- son gain moyen est positif ;
- l'intervalle apparié par date n'est pas franchement défavorable ;
- elle gagne au moins cinq folds sur huit ;
- son éventuel gain ne vient pas seulement d'un changement massif du taux de
  prédictions positives.

La PCA et les facteurs appris sont volontairement repoussés : ils ajoutent des
choix de dimension, d'imputation et de régularisation qui n'ont de sens que si
la neutralisation simple montre déjà un signal.
