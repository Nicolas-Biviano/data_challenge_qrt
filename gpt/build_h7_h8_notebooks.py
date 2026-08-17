"""Build the two presentation notebooks for H7 and H8."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def base_notebook() -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    return notebook


h7 = base_notebook()
h7["cells"] = [
    md(
        """
# H7 — sélection stable de transformations centrées sur zéro

## Question

Des transformations qui dilatent les petits returns peuvent-elles améliorer la zone où
le score de V2 est proche de zéro ? Pour limiter la variance, 40 transformations
atomiques sont mises en compétition par stability selection, entièrement à l'intérieur
de chaque fold externe.

Le protocole fixe huit sous-échantillonnages de dates, une Elastic Net forte et au plus
quatre features retenues, avec un seul représentant par lag.
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
if (cwd / "gpt" / "outputs" / "h7_zero_focused_stability").exists():
    repo = cwd
elif (cwd.parent / "outputs" / "h7_zero_focused_stability").exists():
    repo = cwd.parent.parent
else:
    raise FileNotFoundError("Lancer depuis la racine ou gpt/notebooks.")
out = repo / "gpt" / "outputs" / "h7_zero_focused_stability"
frequency = pd.read_csv(out / "selection_frequency.csv")
folds = pd.read_csv(out / "fold_comparison.csv")
uncertainty = pd.read_csv(out / "uncertainty.csv")
summary = json.loads((out / "summary.json").read_text())
summary["metrics"]
"""
    ),
    md(
        """
## 1. La sélection est réellement stable

La ligne pointillée représente le seuil pré-enregistré de 50 %. Une forme `tanh` de
`RET_1` et la racine signée de `RET_2` sont sélectionnées dans presque tous les
sous-échantillons. `RET_18` est également fréquent.
"""
    ),
    code(
        """
top = frequency.head(12).sort_values("selection_frequency")
labels = (top.candidate.str.replace("__zero_", " · ", regex=False)
          .str.replace("signed_sqrt", "signed sqrt", regex=False))
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(labels, 100*top.selection_frequency,
               color=np.where(top.selection_frequency.ge(.5), "#4472C4", "#A5A5A5"))
ax.axvline(50, color="#C00000", ls="--", lw=1.5, label="seuil de sélection")
ax.set_xlabel("fréquence parmi les 64 ajustements internes (%)")
ax.set_title("Quelques formes zero-focused sont très reproductibles")
ax.legend()
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
## 2. Une sélection stable n'implique pas un gain incrémental

Les features sélectionnées restent très corrélées aux returns bruts déjà présents dans
V2. Le signe du gain change selon le fold, malgré une sélection similaire.
"""
    ),
    code(
        """
fig, ax = plt.subplots(figsize=(10, 5))
colors = np.where(folds.gain_vs_v2.ge(0), "#70AD47", "#C0504D")
bars = ax.bar(folds.outer_fold.astype(int), 100*folds.gain_vs_v2, color=colors)
ax.axhline(0, color="black", lw=1)
ax.set_xticks(range(1, 9))
ax.set_xlabel("fold externe")
ax.set_ylabel("gain H7 − V2 (points d'accuracy)")
ax.set_title("H7 ne gagne que dans 4 folds sur 8")
plt.show()
folds[["outer_fold", "selected_features", "accuracy", "v2_accuracy", "gain_vs_v2"]]
"""
    ),
    md(
        """
## 3. Résultat global avec incertitude

H7 atteint **52,452 %**, contre **52,500 %** pour V2. Le gain est de **−0,048 point**
avec un intervalle bootstrap par date **[−0,120 ; +0,021] point**.
"""
    ),
    code(
        """
h7_row = uncertainty[uncertainty.model.eq("H7 zero stability")].iloc[0]
fig, ax = plt.subplots(figsize=(7, 4.5))
mean = 100*h7_row.gain_vs_v2
lower = 100*(h7_row.gain_vs_v2-h7_row.gain_ci_low)
upper = 100*(h7_row.gain_ci_high-h7_row.gain_vs_v2)
ax.errorbar([0], [mean], yerr=[[lower], [upper]], fmt="o", markersize=9,
            capsize=8, color="#4472C4")
ax.axhline(0, color="black", ls="--", lw=1)
ax.set_xlim(-.7, .7); ax.set_xticks([0], ["H7 − V2"])
ax.set_ylabel("gain (points d'accuracy)")
ax.set_title("Le bootstrap n'identifie aucun gain")
plt.show()
uncertainty.round(6)
"""
    ),
    md(
        """
## 4. La zone proche de zéro ne s'améliore pas

Dans la moitié des lignes au plus faible margin V2, H7 passe de 51,008 % à 50,919 %.
Étirer les entrées près de zéro ne récupère donc pas le signal manquant sous cette forme.
"""
    ),
    code(
        """
low = summary["metrics"]["low_margin"]
fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(["V2", "H7"], [100*low["v2_accuracy"], 100*low["h7_accuracy"]],
              color=["#A5A5A5", "#4472C4"])
ax.bar_label(bars, fmt="%.3f %%", padding=4)
ax.set_ylim(50.5, 51.2)
ax.set_ylabel("accuracy faible margin (%)")
ax.set_title("Pas de récupération du signal autour de zéro")
plt.show()
"""
    ),
    md(
        """
## Conclusion

- La stability selection trouve des coefficients très reproductibles.
- Ces coefficients représentent surtout une autre paramétrisation des returns déjà
  présents, pas un signal résiduel.
- Accuracy H7 : 0,524519 ; gain : −0,000482 ; gate échoué.
- Aucune soumission et aucun élargissement de la banque de transformations.
"""
    ),
]


h8 = base_notebook()
h8["cells"] = [
    md(
        """
# H8 — les relations feature–target changent-elles réellement de signe ?

## Deux questions séparées

1. Les variations de pente entre dates sont-elles reproductibles ou seulement du bruit ?
2. Si elles sont réelles, peut-on les prédire à partir d'un profil de date X-only ?

Pour chaque return, la pente est la corrélation cross-sectionnelle avec la target
continue dans une date, transformée par Fisher puis rétractée par un modèle à effets
aléatoires. Les allocations de chaque date sont divisées en deux moitiés disjointes pour
le test de reproductibilité.
"""
    ),
    code(
        """
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.figsize": (10, 5.3), "axes.titlesize": 14,
                     "axes.labelsize": 11, "legend.frameon": False})
cwd = Path.cwd().resolve()
if (cwd / "gpt" / "outputs" / "h8_slope_inversion").exists():
    repo = cwd
elif (cwd.parent / "outputs" / "h8_slope_inversion").exists():
    repo = cwd.parent.parent
else:
    raise FileNotFoundError("Lancer depuis la racine ou gpt/notebooks.")
out = repo / "gpt" / "outputs" / "h8_slope_inversion"
slopes = pd.read_csv(out / "date_slopes.csv")
reliability = pd.read_csv(out / "split_half_reliability.csv")
predictions = pd.read_csv(out / "oof_slope_predictions.csv")
metrics = pd.read_csv(out / "feature_metrics.csv")
gate = pd.read_csv(out / "gate.csv")
reliability.round(4)
"""
    ),
    md(
        """
## 1. Oui : les variations de pente sont fortement reproductibles

Les corrélations entre deux moitiés indépendantes sont comprises entre 0,757 et 0,793.
L'accord de signe est proche de 80 %, et dépasse 93 % lorsque les deux estimations sont
supérieures à une erreur standard.
"""
    ),
    code(
        """
ordered = reliability.sort_values("split_half_pearson")
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(ordered)); width=.36
ax.bar(x-width/2, 100*ordered.split_half_pearson, width, label="corrélation des pentes",
       color="#4472C4")
ax.bar(x+width/2, 100*ordered.split_half_sign_agreement, width, label="accord de signe",
       color="#70AD47")
ax.set_xticks(x, ordered.feature)
ax.set_ylim(70, 85)
ax.set_ylabel("%")
ax.set_title("Deux moitiés d'allocations retrouvent le même régime")
ax.legend()
plt.show()
"""
    ),
    code(
        """
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
for ax, feature in zip(axes, ["RET_1", "RET_18"]):
    d = slopes[slopes.feature.eq(feature)]
    ax.scatter(d.z_half_0, d.z_half_1, s=9, alpha=.28, color="#4472C4")
    limit = np.nanquantile(np.abs(d[["z_half_0", "z_half_1"]]), .99)
    ax.plot([-limit, limit], [-limit, limit], color="black", ls="--", lw=1)
    ax.axhline(0, color="grey", lw=.8); ax.axvline(0, color="grey", lw=.8)
    ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit)
    ax.set_title(feature)
    ax.set_xlabel("pente moitié 1")
axes[0].set_ylabel("pente moitié 2")
fig.suptitle("La variation par date se réplique sur des allocations disjointes", y=1.02)
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
## 2. La relation globale masque fréquemment deux signes opposés

`RET_1` conserve un effet global positif plus visible. Pour les autres returns, la pente
globale est presque nulle alors que 46 à 51 % des dates présentent le signe opposé.
"""
    ),
    code(
        """
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
axes[0].bar(reliability.feature, 100*reliability.global_correlation, color="#4472C4")
axes[0].axhline(0, color="black", lw=1)
axes[0].set_ylabel("corrélation globale (%)")
axes[0].set_title("Relation moyenne")
axes[1].bar(reliability.feature, 100*reliability.shrunk_inversion_rate, color="#ED7D31")
axes[1].axhline(50, color="black", ls="--", lw=1)
axes[1].set_ylabel("dates de signe opposé (%)")
axes[1].set_title("Inversions après shrinkage")
fig.suptitle("Une moyenne faible cache des régimes opposés", y=1.02)
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
## 3. Mais le profil X-only actuel ne sait pas prévoir le régime

Le Ridge utilise les moments, missingness et compositions de groupe du profil H5. Toutes
les corrélations pente prédite/réelle restent sous 0,064 et tous les R² sont négatifs.
La meilleure AUC d'inversion est seulement 0,528.
"""
    ),
    code(
        """
ordered = metrics.sort_values("correlation")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
axes[0].bar(ordered.feature, ordered.correlation, color="#4472C4")
axes[0].axhline(.10, color="#C00000", ls="--", label="gate 0,10")
axes[0].axhline(0, color="black", lw=1)
axes[0].set_title("Prédiction de la pente")
axes[0].set_ylabel("corrélation OOF")
axes[0].legend()
axes[1].bar(ordered.feature, ordered.inversion_auc, color="#ED7D31")
axes[1].axhline(.55, color="#C00000", ls="--", label="gate 0,55")
axes[1].axhline(.5, color="black", lw=1)
axes[1].set_title("Prédiction de l'inversion")
axes[1].set_ylabel("ROC-AUC OOF")
axes[1].legend()
fig.suptitle("Aucune des huit features ne franchit le gate", y=1.02)
plt.tight_layout()
plt.show()
metrics.round(4)
"""
    ),
    code(
        """
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
for ax, feature in zip(axes, ["RET_1", "RET_18"]):
    d = predictions[predictions.feature.eq(feature)]
    colors = np.where(d.inversion, "#C0504D", "#5B9BD5")
    ax.scatter(d.actual_shrunk_z, d.predicted_shrunk_z, s=10, alpha=.32, c=colors)
    limit = np.quantile(np.abs(d.actual_shrunk_z), .99)
    ax.plot([-limit, limit], [-limit, limit], color="black", ls="--", lw=1)
    ax.axhline(0, color="grey", lw=.8); ax.axvline(0, color="grey", lw=.8)
    ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit)
    ax.set_title(feature)
    ax.set_xlabel("pente réelle rétractée")
axes[0].set_ylabel("pente prédite X-only")
fig.suptitle("Le prédicteur reste proche de la moyenne globale", y=1.02)
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
## Conclusion

- **Découverte forte :** les inversions de pente sont réelles, avec une reproductibilité
  split-half très élevée.
- **Échec actuel :** les moments marginaux du profil H5 ne prédisent pas ces inversions.
- **Décision :** aucun modèle adaptatif et aucune soumission H8.

La prochaine hypothèse doit décrire la géométrie multivariée de la date avec des
corrélations entre lags, une structure factorielle cross-sectionnelle et des relations
returns–volumes. Une pente est une covariance : il est cohérent que de simples moyennes
et dispersions ne suffisent pas à prévoir son signe.
"""
    ),
]


outputs = [
    (h7, ROOT / "notebooks" / "H7_zero_focused_stability_selection.ipynb"),
    (h8, ROOT / "notebooks" / "H8_predictable_slope_inversions.ipynb"),
]
for notebook, path in outputs:
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)
    print(path)

