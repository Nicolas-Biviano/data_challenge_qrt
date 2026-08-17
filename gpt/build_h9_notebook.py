"""Build the H9 allocation-archetype and relational-regime notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "notebooks" / "H9_allocation_archetypes_and_relational_regimes.ipynb"


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
        """
# H9 — donner un comportement aux allocations pour comprendre les régimes

## Idée

Les noms des allocations sont anonymes, mais leurs distributions X-only permettent de
leur attribuer des archétypes descriptifs : faible ou forte volatilité, turnover,
disponibilité du volume, alignement au facteur commun, persistance ou opposition des
lags.

Ces archétypes ne prétendent pas révéler l'identité réelle des allocations. Ils servent
à construire un vocabulaire économique prudent et à décrire chaque date par les
relations entre ses composantes, plutôt que par de simples moyennes.
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
plt.rcParams.update({"figure.figsize": (10, 5.3), "axes.titlesize": 14,
                     "axes.labelsize": 11, "legend.frameon": False})
cwd = Path.cwd().resolve()
if (cwd / "gpt" / "outputs" / "h9_allocation_archetypes_relational").exists():
    repo = cwd
elif (cwd.parent / "outputs" / "h9_allocation_archetypes_relational").exists():
    repo = cwd.parent.parent
else:
    raise FileNotFoundError("Lancer depuis la racine ou gpt/notebooks.")
out = repo / "gpt" / "outputs" / "h9_allocation_archetypes_relational"
membership = pd.read_csv(out / "global_archetype_membership.csv")
centroids = pd.read_csv(out / "archetype_centroids_z.csv", index_col=0)
stability = pd.read_csv(out / "archetype_stability.csv")
fold_archetypes = pd.read_csv(out / "archetype_fold_diagnostics.csv")
folds = pd.read_csv(out / "fold_metrics.csv")
metrics = pd.read_csv(out / "feature_metrics.csv")
comparison = pd.read_csv(out / "h8_h9_comparison.csv")
gate = pd.read_csv(out / "gate.csv")
summary = json.loads((out / "summary.json").read_text())
summary
"""
    ),
    md(
        """
## 1. Les archétypes sont stables et ne recopient pas `GROUP`

- Adjusted Rand Index moyen entre les partitions des huit folds : **0,878**.
- Adjusted Mutual Information moyenne avec `GROUP` : **0,126**.
- Les 278 allocations sont observées dans chaque train externe : aucun `UNKNOWN`.

Le comportement X-only apporte donc une segmentation différente du groupe fourni.
"""
    ),
    code(
        """
counts = membership.groupby(["archetype", "GROUP"]).size().unstack(fill_value=0)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
total = counts.sum(axis=1)
size_bars = axes[0].bar(total.index.astype(str), total, color="#4472C4")
axes[0].bar_label(size_bars, padding=3)
axes[0].set_xlabel("archétype")
axes[0].set_ylabel("nombre d'allocations")
axes[0].set_title("Taille des archétypes")
normalized = counts.div(counts.sum(axis=1), axis=0)
bottom = np.zeros(len(normalized))
for group in normalized.columns:
    axes[1].bar(normalized.index.astype(str), 100*normalized[group], bottom=bottom,
                label=f"GROUP {group}")
    bottom += 100*normalized[group].to_numpy()
axes[1].set_xlabel("archétype")
axes[1].set_ylabel("composition (%)")
axes[1].set_title("Chaque archétype mélange plusieurs GROUP")
axes[1].legend(ncol=2)
plt.tight_layout()
plt.show()
counts
"""
    ),
    md(
        """
## 2. Lecture comportementale prudente

Les centroïdes sont exprimés en écarts-types par rapport à l'allocation moyenne. La
figure sélectionne quelques dimensions interprétables. Les clusters 3 à 5 sont petits :
ils représentent davantage des comportements extrêmes que de grandes familles.
"""
    ),
    code(
        """
selected = [
    "RET_1_std", "RET_1_abs_mean", "ret1_ret2_corr", "ret1_date_factor_corr",
    "ret1_sv1_corr", "turnover_rank_mean", "sv1_observed_rate",
    "volume_observed_fraction",
]
matrix = centroids[selected]
fig, ax = plt.subplots(figsize=(11, 5.5))
image = ax.imshow(matrix, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
ax.set_yticks(range(len(matrix)), [f"archétype {i}" for i in matrix.index])
ax.set_xticks(range(len(selected)), [x.replace("_", " ") for x in selected], rotation=45,
              ha="right")
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        value = matrix.iloc[i,j]
        ax.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=8,
                color="white" if abs(value) > 1.5 else "black")
fig.colorbar(image, ax=ax, label="centroïde standardisé")
ax.set_title("Comportements X-only des six archétypes")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
Interprétation descriptive :

- **0 — faible volatilité / lags persistants / volume aligné** ;
- **1 — forte volatilité / forte exposition au facteur commun** ;
- **2 — lags plutôt opposés / volume moins disponible** ;
- **3 — très liquide et high-turnover** ;
- **4 — deux outliers de volatilité extrême** ;
- **5 — turnover extrême et faible alignement au facteur**.

« Persistant » décrit uniquement les relations entre returns passés disponibles, pas une
capacité démontrée à prédire la target future.
"""
    ),
    md(
        """
## 3. Le profil relationnel sur-apprend fortement

Le profil final contient 290 variables pour 2 522 dates. Malgré Ridge `alpha=100` et une
logistique `C=0.01`, le contraste train-validation est très grand.
"""
    ),
    code(
        """
mean_folds = folds.groupby("feature")[[
    "ridge_train_correlation", "ridge_valid_correlation",
    "logit_train_auc", "logit_valid_auc"
]].mean().loc[metrics.feature]
x = np.arange(len(mean_folds)); width=.36
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
axes[0].bar(x-width/2, mean_folds.ridge_train_correlation, width, label="train",
            color="#A5A5A5")
axes[0].bar(x+width/2, mean_folds.ridge_valid_correlation, width, label="validation",
            color="#4472C4")
axes[0].axhline(0, color="black", lw=1)
axes[0].set_xticks(x, mean_folds.index)
axes[0].set_ylabel("corrélation")
axes[0].set_title("Pente continue Ridge")
axes[1].bar(x-width/2, mean_folds.logit_train_auc, width, label="train",
            color="#A5A5A5")
axes[1].bar(x+width/2, mean_folds.logit_valid_auc, width, label="validation",
            color="#ED7D31")
axes[1].axhline(.5, color="black", ls="--", lw=1)
axes[1].set_xticks(x, mean_folds.index)
axes[1].set_ylabel("ROC-AUC")
axes[1].set_title("Inversion directe")
axes[1].legend()
fig.suptitle("Le signal apparent au train disparaît hors date", y=1.02)
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
## 4. Comparaison au profil marginal H8

`RET_4` et `RET_7` progressent en AUC, mais aucun résultat n'approche le gate 0,55 avec
une généralisation suffisante. Pour la pente continue, H9 ne dépasse jamais 0,067 de
corrélation et reste souvent inférieur à H8.
"""
    ),
    code(
        """
order = comparison.sort_values("logit_valid_auc").feature
d = comparison.set_index("feature").loc[order]
x = np.arange(len(d)); width=.36
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
axes[0].bar(x-width/2, d.h8_marginal_ridge_correlation, width, label="H8 marginal",
            color="#A5A5A5")
axes[0].bar(x+width/2, d.ridge_correlation, width, label="H9 relationnel",
            color="#4472C4")
axes[0].axhline(.10, color="#C00000", ls="--", label="gate")
axes[0].axhline(0, color="black", lw=1)
axes[0].set_xticks(x, d.index)
axes[0].set_ylabel("corrélation OOF")
axes[0].set_title("Pente continue")
axes[1].bar(x-width/2, d.h8_marginal_inversion_auc, width, label="H8 marginal",
            color="#A5A5A5")
axes[1].bar(x+width/2, d.logit_valid_auc, width, label="H9 relationnel direct",
            color="#ED7D31")
axes[1].axhline(.55, color="#C00000", ls="--", label="gate")
axes[1].axhline(.5, color="black", lw=1)
axes[1].set_xticks(x, d.index)
axes[1].set_ylabel("ROC-AUC OOF")
axes[1].set_title("Inversion")
axes[1].legend()
fig.suptitle("Les gains relationnels sont faibles et irréguliers", y=1.02)
plt.tight_layout()
plt.show()
gate
"""
    ),
    md(
        """
## Conclusion

- Les archétypes sont stables et apportent un vocabulaire différent de `GROUP`.
- Le KMeans isole toutefois plusieurs petits groupes d'outliers.
- Les 290 caractéristiques relationnelles provoquent un overfit manifeste.
- Aucun gate H9 n'est passé ; aucune pente adaptative et aucune soumission.

La prochaine expérience éventuelle doit réduire la variance : clustering robuste aux
outliers et sélection **imbriquée** de régularisations beaucoup plus fortes. Ajouter
encore des covariances ou un modèle plus flexible serait prématuré.
"""
    ),
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUTPUT)
print(OUTPUT)
