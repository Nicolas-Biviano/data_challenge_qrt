"""Build the H5 notebook from the frozen test-sized panel outputs."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "notebooks" / "H5_test_sized_panels_and_confidence.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}

nb["cells"] = [
    markdown(
        """
# H5 — panels de 120 dates et confiance portée par l'amplitude

## Question

Notre CV moyenne couvre beaucoup plus de dates que le test public, qui n'en contient que
120. Peut-on construire un estimateur local plus proche du test, puis vérifier si
`|rendement prédit|` indique réellement quand le signe est fiable ?

Ce notebook présente une expérience **figée avant lecture des résultats** : aucun modèle,
seuil ou feature n'est choisi ici. Les prédictions OOF de V1 Ridge, V2 logistique et
LightGBM linéaire sont réutilisées telles quelles.

### Résultat en une phrase

L'appariement X-only améliore nettement la ressemblance avec le test, mais le classement
des modèles reste trop incertain pour tuner des millièmes. En revanche, la précision
augmente fortement et dans les 8 folds avec l'amplitude du score prédit.
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
plt.rcParams.update({
    "figure.figsize": (10, 5.5),
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "legend.frameon": False,
})

cwd = Path.cwd().resolve()
if (cwd / "gpt" / "outputs" / "test_sized_panels").exists():
    repo = cwd
elif (cwd.parent / "outputs" / "test_sized_panels").exists():
    repo = cwd.parent.parent
else:
    raise FileNotFoundError("Lancer le notebook depuis la racine du dépôt ou gpt/notebooks.")

out = repo / "gpt" / "outputs" / "test_sized_panels"
balance = pd.read_csv(out / "panel_balance.csv")
metrics = pd.read_csv(out / "panel_metrics.csv")
panel_summary = pd.read_csv(out / "panel_summary.csv")
paired = pd.read_csv(out / "paired_summary.csv")
confidence = pd.read_csv(out / "confidence_curve.csv")
confidence_folds = pd.read_csv(out / "confidence_folds.csv")
summary = json.loads((out / "summary.json").read_text())

COLORS = {"V1 Ridge": "#4472C4", "V2 logistic": "#70AD47", "LightGBM linear": "#ED7D31"}
REGIME_COLORS = {"uniform": "#A5A5A5", "matched": "#5B9BD5"}
print(f"{summary['n_panels_per_regime']:,} panels par régime, "
      f"{summary['n_test_dates_per_panel']} dates par panel")
"""
    ),
    markdown(
        """
## 1. Construction de l'estimateur

Chaque date est décrite uniquement avec `X` : nombre de lignes et d'allocations,
moments des returns/volumes/turnover, missingness des volumes et composition en `GROUP`.
Après imputation, standardisation et réduction à 15 composantes principales, chacune des
120 dates test est appariée à ses dates train voisines.

On compare :

- **1 000 panels appariés** de 120 dates train uniques, tirées près des dates test ;
- **1 000 panels uniformes** de 120 dates train, qui servent de contrôle.

Cette construction est transductive, mais strictement X-only. Elle ne corrige pas un
éventuel changement de relation entre `X` et `y`.
"""
    ),
    code(
        """
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
plots = [
    ("profile_pca_mean_distance", "Distance du profil moyen au test"),
    ("profile_mean_abs_smd", "Écart standardisé moyen au test"),
]
for ax, (col, title) in zip(axes, plots):
    for regime in ["uniform", "matched"]:
        values = balance.loc[balance.regime.eq(regime), col]
        ax.hist(values, bins=35, density=True, alpha=0.55,
                label=regime, color=REGIME_COLORS[regime])
        ax.axvline(values.mean(), color=REGIME_COLORS[regime], lw=2)
    ax.set_title(title)
    ax.set_xlabel("plus bas = plus proche")
    ax.set_ylabel("densité")
axes[0].legend(title="Panel")
fig.suptitle("L'appariement rapproche effectivement les panels du test", y=1.03, fontsize=15)
plt.tight_layout()
plt.show()

balance.groupby("regime")[["profile_pca_mean_distance", "profile_mean_abs_smd",
                           "profile_max_abs_smd"]].mean().round(4)
"""
    ),
    markdown(
        """
**Lecture.** La distance moyenne PCA passe de **2,174** à **1,010**, et l'écart
standardisé moyen de **0,216** à **0,118**. Le mécanisme fait donc bien ce qui était
prévu : il produit des échantillons train plus ressemblants au test selon `X`.
"""
    ),
    markdown(
        """
## 2. Distribution des scores sur des tests de même taille

Une valeur ponctuelle masque l'incertitude due au choix des 120 dates. Les distributions
ci-dessous montrent l'accuracy pondérée par ligne sur chaque panel.
"""
    ),
    code(
        """
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, regime in zip(axes, ["uniform", "matched"]):
    for model in ["V1 Ridge", "V2 logistic", "LightGBM linear"]:
        values = metrics.loc[(metrics.regime.eq(regime)) & (metrics.model.eq(model)), "accuracy"]
        ax.hist(values, bins=35, density=True, histtype="step", lw=2,
                color=COLORS[model], label=model)
        ax.axvline(values.mean(), color=COLORS[model], lw=1.2, alpha=.8)
    ax.set_title("Panels " + ("uniformes" if regime == "uniform" else "appariés au test"))
    ax.set_xlabel("accuracy sur 120 dates")
    ax.axvline(.5, color="black", ls=":", lw=1)
axes[0].set_ylabel("densité")
axes[1].legend(loc="upper left")
fig.suptitle("Les distributions des trois modèles se recouvrent presque totalement", y=1.02)
plt.tight_layout()
plt.show()

panel_summary.assign(
    mean=lambda d: d["mean"].map(lambda x: f"{100*x:.3f} %"),
    interval_95=lambda d: d.apply(lambda r: f"[{100*r.q025:.3f} ; {100*r.q975:.3f}] %", axis=1),
)[["regime", "model", "mean", "interval_95"]]
"""
    ),
    markdown(
        """
Sur les panels appariés, LightGBM a la meilleure moyenne (**51,925 %**), puis V1
(**51,862 %**) et V2 (**51,859 %**). L'ordre correspond au leaderboard connu, mais
l'écart LightGBM − V1 n'est que de **0,062 point**, face à une variabilité de plusieurs
dixièmes de point.
"""
    ),
    markdown(
        """
## 3. Comparaison appariée LightGBM − V1

Comparer les deux modèles sur exactement les mêmes dates élimine une partie du bruit.
La ligne orange indique l'écart public observé : seulement **+0,0188 point** en faveur
de LightGBM.
"""
    ),
    code(
        """
pivot = metrics.pivot_table(index=["regime", "panel"], columns="model", values="accuracy")
public_delta = 0.5120175713837465 - 0.5118293065578914

fig, ax = plt.subplots(figsize=(10, 5.2))
for regime in ["uniform", "matched"]:
    delta = pivot.loc[regime, "LightGBM linear"] - pivot.loc[regime, "V1 Ridge"]
    ax.hist(100 * delta, bins=40, density=True, alpha=.48,
            color=REGIME_COLORS[regime], label=regime)
    ax.axvline(100 * delta.mean(), color=REGIME_COLORS[regime], lw=2)
ax.axvline(0, color="black", ls="--", lw=1.4, label="égalité")
ax.axvline(100 * public_delta, color="#ED7D31", ls=":", lw=2.5,
           label="écart public observé")
ax.set_xlabel("LightGBM − V1 (points d'accuracy)")
ax.set_ylabel("densité")
ax.set_title("Même après appariement, le signe de l'écart reste incertain")
ax.legend()
plt.show()

paired.loc[paired.comparison.str.contains("LightGBM")].assign(
    mean_pp=lambda d: 100*d["mean"],
    interval_95_pp=lambda d: d.apply(lambda r: f"[{100*r.q025:.3f} ; {100*r.q975:.3f}]", axis=1),
)[["regime", "mean_pp", "interval_95_pp", "probability_positive"]].round(4)
"""
    ),
    markdown(
        """
Le panel apparié donne LightGBM gagnant dans **57,6 %** des tirages. Son intervalle
empirique à 95 % pour le gain est **[-0,540 ; +0,632] point** : immensément plus large
que l'écart public. Le classement public exact apparaît dans **25,3 %** des panels
appariés contre **14,9 %** des panels uniformes.

**Décision :** l'estimateur apparié est meilleur comme *stress test*, mais pas assez
précis pour choisir entre deux modèles séparés par quelques centièmes de point.
"""
    ),
    code(
        """
rank_rows = paired[paired.comparison.eq("classement public exact")].copy()
fig, ax = plt.subplots(figsize=(7.5, 4.5))
bars = ax.bar(rank_rows.regime, 100*rank_rows.probability_positive,
              color=[REGIME_COLORS[x] for x in rank_rows.regime])
ax.bar_label(bars, fmt="%.1f %%", padding=4)
ax.set_ylim(0, 30)
ax.set_ylabel("fréquence")
ax.set_title("Fréquence du classement public exact\\nLightGBM > V1 > V2")
plt.show()
"""
    ),
    markdown(
        """
## 4. L'amplitude du rendement prédit porte-t-elle de la confiance ?

Dans chaque fold séparément, on classe les observations par percentile de `|score|`.
Cette normalisation évite qu'une simple différence d'échelle entre folds explique la
courbe. Aucun seuil n'est appris.
"""
    ),
    code(
        """
fig, ax = plt.subplots(figsize=(10, 5.4))
for model in ["V1 Ridge", "LightGBM linear"]:
    d = confidence[confidence.model.eq(model)]
    ax.plot(d.confidence_decile, 100*d.accuracy, marker="o", lw=2.5,
            color=COLORS[model], label=model)
ax.axhline(50, color="black", ls=":", lw=1)
ax.set_xticks(range(1, 11))
ax.set_xlabel("décile de confiance dans le fold (1 = |score| faible)")
ax.set_ylabel("accuracy (%)")
ax.set_title("Plus |score| est grand, plus le signe est fiable")
ax.legend()
plt.show()

confidence.pivot(index="confidence_decile", columns="model", values="accuracy").mul(100).round(2)
"""
    ),
    markdown(
        """
Pour V1, l'accuracy monte de **50,38 %** dans le premier décile à **57,24 %** dans le
dernier. Pour LightGBM, elle passe de **49,91 %** à **56,92 %**. La valeur absolue de la
cible elle-même augmente aussi dans les déciles élevés : les modèles identifient en
partie les mouvements de plus grande amplitude, dont le signe est plus prévisible.
"""
    ),
    code(
        """
fold_plot = confidence_folds.copy()
fold_plot["label"] = "fold " + fold_plot.fold.astype(str)
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(8)
width = .36
for offset, model in [(-width/2, "V1 Ridge"), (width/2, "LightGBM linear")]:
    d = fold_plot[fold_plot.model.eq(model)].sort_values("fold")
    ax.bar(x + offset, 100*d.top_minus_bottom, width=width,
           color=COLORS[model], label=model)
ax.axhline(0, color="black", lw=1)
ax.set_xticks(x, [f"fold {i}" for i in range(1, 9)])
ax.set_ylabel("top décile − bottom décile (points)")
ax.set_title("Le gradient de confiance est positif dans les 8 folds")
ax.legend()
plt.show()
"""
    ),
    markdown(
        """
## 5. Peut-on déjà choisir entre V1 et LightGBM avec cette confiance ?

Un diagnostic simple choisit le modèle au percentile de `|score|` le plus élevé quand
leurs signes divergent. Il obtient **52,524 % OOF**, contre **52,471 %** pour V1 : un
gain de seulement **+0,053 point**. Sur les désaccords, le choix est correct à
**50,42 %**, quasiment le hasard.

Cette expérience sépare deux affirmations :

1. `|score|` estime bien la difficulté absolue d'une ligne — **validé** ;
2. `|score|` sait quel modèle écouter lors d'un désaccord — **non démontré**.

Le sélecteur ne doit donc pas être soumis.
"""
    ),
    code(
        """
s = summary["confidence_selector"]
pd.DataFrame({
    "mesure": ["V1", "LightGBM", "sélecteur de confiance", "accuracy sur désaccords",
                "part des désaccords"],
    "valeur": [s["v1_accuracy"], s["lgbm_accuracy"], s["overall_accuracy"],
               s["disagreement_accuracy"], s["disagreement_rate"]],
}).assign(valeur=lambda d: d.valeur.map(lambda x: f"{100*x:.3f} %"))
"""
    ),
    markdown(
        """
## 6. Conclusion et prochaine expérience autorisée

- Les panels appariés sont utiles pour mesurer la **fragilité au choix des dates**, pas
  pour optimiser des gains minuscules.
- L'amplitude du score est une mesure OOF robuste de **confiance absolue**.
- Elle ne résout pas encore la sélection entre modèles et ne justifie aucune soumission.

La prochaine expérience rationnelle consiste à améliorer l'estimation de l'erreur
conditionnelle : prédire hors fold `P(erreur | X, |score|, désaccord, régime de marché)`
avec un modèle très régularisé, puis l'évaluer par log-loss/Brier et calibration, avant
toute utilisation pour pondérer un ensemble. Les panels appariés serviront uniquement de
stress test secondaire.
"""
    ),
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUTPUT)
print(OUTPUT)
