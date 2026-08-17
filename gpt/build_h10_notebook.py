"""Build the pre-executed H10 defactored-returns notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "notebooks" / "H10_defactored_returns.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
nb["cells"] = [
    md(
        r"""
# H10 — les rendements « dé-factorisés » sont-ils plus prévisibles ?

L'idée testée est de séparer chaque rendement passé en mouvement commun de la
date, mouvement relatif du `GROUP`, et mouvement propre à l'allocation :

\[
RET_{i,t-h}=M_{t-h}+G_{g(i),t-h}+\varepsilon_{i,t-h}.
\]

Toutes les moyennes sont leave-one-out et construites uniquement avec `X`.
Les dates de validation restent entièrement hors entraînement, aucun target
encoding n'est utilisé, et aucun ordre des dates anonymisées n'est supposé.

Cette expérience porte exclusivement sur `RET_*`. `SIGNED_VOLUME_*` n'est pas
le volume d'un titre et n'est pas utilisé ici.
"""
    ),
    code(
        """
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.figsize": (10, 5.2), "axes.titlesize": 14,
                     "axes.labelsize": 11, "legend.frameon": False})
cwd = Path.cwd().resolve()
if (cwd / "gpt" / "outputs" / "h10_defactored_returns").exists():
    repo = cwd
elif (cwd.parent / "outputs" / "h10_defactored_returns").exists():
    repo = cwd.parent.parent
else:
    raise FileNotFoundError("Lancer depuis la racine ou gpt/notebooks.")
out = repo / "gpt" / "outputs" / "h10_defactored_returns"
results = pd.read_csv(out / "results.csv")
uncertainty = pd.read_csv(out / "paired_uncertainty.csv")
folds = pd.read_csv(out / "fold_metrics.csv")
components = pd.read_csv(out / "component_diagnostics.csv")
summary = json.loads((out / "summary.json").read_text())
summary["best_eligible_experiment"], summary["pca_authorized"]
"""
    ),
    md(
        r"""
## 1. Construction exacte

Pour une ligne (i), une date (t) et un lag (h) :

\[
\begin{aligned}
M^{(-i)}_{t,h} &= \operatorname{mean}_{j\ne i}(RET_{j,t-h}),\\
G^{(-i)}_{g,t,h} &=
\operatorname{mean}_{j\ne i,g(j)=g(i)}(RET_{j,t-h})-M^{(-i)}_{t,h},\\
\varepsilon^{(-i)}_{i,t,h} &=
RET_{i,t-h}-\operatorname{mean}_{j\ne i,g(j)=g(i)}(RET_{j,t-h}).
\end{aligned}
\]

Il y a environ 209 allocations par date et 69 par `date × GROUP`. La
neutralisation ne repose donc pas sur des groupes minuscules.
"""
    ),
    md(
        """
## 2. La majorité de la variance est spécifique, mais le commun n'est pas nul

Le graphique donne l'écart-type de chaque composante divisé par celui du
rendement brut. Ces ratios ne s'additionnent pas : les composantes sont
corrélées et la variance n'est pas additive en écart-type.
"""
    ),
    code(
        """
labels = {"market": "marché", "group_tilt": "écart GROUP",
          "group_resid": "résidu propre", "market_resid": "résidu au marché"}
pivot = components.pivot(index="lag", columns="component", values="std_over_raw")
order = [f"RET_{lag}" for lag in [1,2,3,4,7,8,9,18]]
pivot = pivot.loc[order]
fig, ax = plt.subplots(figsize=(11, 5.4))
x = np.arange(len(pivot)); width = 0.19
colors = ["#4472C4", "#ED7D31", "#70AD47", "#A5A5A5"]
for position, component in enumerate(["market", "group_tilt", "group_resid", "market_resid"]):
    ax.bar(x + (position-1.5)*width, pivot[component], width,
           label=labels[component], color=colors[position])
ax.axhline(1, color="black", lw=1, ls="--")
ax.set_xticks(x, pivot.index)
ax.set_ylabel("écart-type / écart-type du RET brut")
ax.set_title("Taille des composantes pour les huit lags de V2")
ax.legend(ncol=4, loc="upper center")
plt.tight_layout()
plt.show()
pivot.round(3)
"""
    ),
    md(
        """
Le facteur de marché représente environ **25,5 %** de l'écart-type brut,
l'écart de groupe **32–35 %**, et le résidu propre conserve **92–93 %**. Une
faible fraction de variance peut néanmoins contenir un signal directionnel
utile : seule la CV peut trancher.
"""
    ),
    md(
        """
## 3. Résultat OOF avec incertitude appariée

Les barres suivantes représentent le gain de correction par rapport à V2. Pour
chaque ligne, on soustrait `correct_V2` à `correct_variante`, puis on calcule
l'erreur standard sur les 2 522 moyennes par date. C'est la bonne incertitude
pour comparer deux modèles sur exactement les mêmes observations.
"""
    ),
    code(
        """
comparison = results.merge(uncertainty, on="experiment")
comparison = comparison[comparison.experiment.ne("baseline_raw")].copy()
pretty = {
    "market_defactored_only": "résidu marché seul",
    "group_defactored_only": "résidu GROUP seul",
    "raw_plus_defactored": "brut + résidus",
    "hierarchical_components": "3 composantes",
    "hierarchical_raw_local_slope": "3 composantes + pente brute",
}
comparison["label"] = comparison.experiment.map(pretty)
comparison = comparison.sort_values("gain_accuracy")
lower = comparison.gain_accuracy - comparison.ci95_low
upper = comparison.ci95_high - comparison.gain_accuracy
fig, ax = plt.subplots(figsize=(10.5, 5.4))
colors = ["#C00000" if high < 0 else "#ED7D31" for high in comparison.ci95_high]
ax.barh(comparison.label, 100*comparison.gain_accuracy, color=colors, alpha=.85)
ax.errorbar(100*comparison.gain_accuracy, comparison.label,
            xerr=np.vstack([100*lower, 100*upper]), fmt="none",
            ecolor="black", capsize=4, lw=1.3)
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("gain d'accuracy vs V2 (points de pourcentage, IC 95 % par date)")
ax.set_title("Aucune neutralisation n'améliore V2")
plt.tight_layout()
plt.show()
comparison[["label", "accuracy", "gain_accuracy", "ci95_low", "ci95_high", "folds_won"]]
"""
    ),
    md(
        """
## 4. Stabilité fold par fold

La neutralisation au marché et au groupe perd dans chacun des huit folds. La
version hiérarchique initiale gagne légèrement deux folds, mais son gain moyen
est négatif. Conserver la pente locale `ALLOCATION × RET_1 brut` ne sauve pas
la décomposition : ce contrôle perd également les huit folds.
"""
    ),
    code(
        """
table = folds.pivot(index="fold", columns="experiment", values="accuracy")
gain = 100*(table.drop(columns="baseline_raw").sub(table.baseline_raw, axis=0))
gain = gain.rename(columns=pretty)
fig, ax = plt.subplots(figsize=(11, 5.2))
image = ax.imshow(gain.T, cmap="RdYlGn", vmin=-.6, vmax=.6, aspect="auto")
ax.set_xticks(range(len(gain)), [f"fold {i}" for i in gain.index])
ax.set_yticks(range(len(gain.columns)), gain.columns)
for row in range(gain.shape[1]):
    for column in range(gain.shape[0]):
        value = gain.iloc[column, row]
        ax.text(column, row, f"{value:+.2f}", ha="center", va="center", fontsize=9)
fig.colorbar(image, ax=ax, label="gain vs V2 (points de %) ")
ax.set_title("Différence d'accuracy dans chaque fold")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
## 5. AUC et taux de prédictions positives

Le rejet ne vient pas uniquement du seuil à 0,5 : les AUC des variantes sont
également inférieures. La neutralisation ne corrige pas non plus le fort taux
de prédictions positives de V2 ; les résidus seuls le font même légèrement
augmenter.
"""
    ),
    code(
        """
auc = folds.groupby("experiment").auc.mean()
diagnostic = results.set_index("experiment")[["accuracy", "positive_prediction_rate"]].join(
    auc.rename("mean_fold_auc")
).loc[["baseline_raw"] + list(pretty)]
diagnostic.index = ["V2 brute"] + [pretty[x] for x in pretty]
diagnostic
"""
    ),
    md(
        """
## Conclusion

Le résultat réfute la version simple de l'hypothèse : **les rendements relatifs
ne sont pas plus prédictifs que les rendements bruts**. Les composantes de
marché et de `GROUP`, bien que plus petites que le résidu propre, contiennent du
signal directionnel utile.

L'ajout de colonnes dé-factorisées crée aussi une représentation très corrélée
sur laquelle la pénalisation L2 change. Le signal nouveau ne compense pas cette
variance supplémentaire. Le gate échoue : pas de PCA, pas de tuning et pas de
soumission H10. Pour les prochaines expériences, nous conservons les returns
bruts et pouvons utiliser les composantes seulement comme diagnostics de
régime.
"""
    ),
    md(
        """
## Reproduction

Depuis la racine du projet :

```bash
.venv/bin/python gpt/h10_defactored_returns.py
.venv/bin/python gpt/build_h10_notebook.py
.venv/bin/python -m jupyter nbconvert --to notebook --execute \\
  --inplace gpt/notebooks/H10_defactored_returns.ipynb
```
"""
    ),
]


OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUTPUT)
print(OUTPUT)
