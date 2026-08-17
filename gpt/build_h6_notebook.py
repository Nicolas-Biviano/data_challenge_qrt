"""Build the executed-results presentation notebook for H6."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "notebooks" / "H6_conditional_error_estimator.ipynb"


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
# H6 — estimer la probabilité d'erreur, puis savoir quand l'utiliser

## Question

L'amplitude du rendement prédit indique qu'une ligne est facile. Peut-on améliorer cette
probabilité avec des features X-only, et cette amélioration permet-elle de choisir entre
V1 et LightGBM ?

Les deux questions sont volontairement séparées : prévoir qu'une ligne est difficile ne
dit pas nécessairement quel modèle sera le moins mauvais.

### Protocole

- prédictions de base déjà OOF ;
- seconde couche cross-fittée sur les mêmes huit folds de dates ;
- aucun target encoding et aucun tuning de seuil ;
- comparaison au taux d'erreur constant ;
- gate de sélection fixé avant calcul : gain dans 7/8 folds et 90 % des panels appariés.
"""
    ),
    code(
        """
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.figsize": (10, 5.5), "axes.titlesize": 14,
                     "axes.labelsize": 11, "legend.frameon": False})

cwd = Path.cwd().resolve()
if (cwd / "gpt" / "outputs" / "h6_error_estimator").exists():
    repo = cwd
elif (cwd.parent / "outputs" / "h6_error_estimator").exists():
    repo = cwd.parent.parent
else:
    raise FileNotFoundError("Lancer depuis la racine du dépôt ou gpt/notebooks.")

out = repo / "gpt" / "outputs" / "h6_error_estimator"
overall = pd.read_csv(out / "error_overall_metrics.csv")
folds = pd.read_csv(out / "error_fold_metrics.csv")
calibration = pd.read_csv(out / "calibration_curve.csv")
abstention = pd.read_csv(out / "abstention_curve.csv")
selectors = pd.read_csv(out / "selector_metrics.csv")
panels = pd.read_csv(out / "panel_selector_summary.csv")
gate = pd.read_csv(out / "selector_gate.csv")
coefficients = pd.read_csv(out / "feature_coefficients_summary.csv")

COLORS = {"constant": "#A5A5A5", "amplitude_logit": "#4472C4",
          "compact_logit": "#ED7D31"}
overall.round(6)
"""
    ),
    md(
        """
## 1. Le compact améliore-t-il réellement la probabilité d'erreur ?

Oui, modestement mais régulièrement. Le Brier et la log-loss sont plus faibles que le
taux constant dans les huit folds pour V1 et LightGBM. L'AUC d'erreur passe d'environ
0,521 avec l'amplitude seule à 0,5235 avec les features compactes.
"""
    ),
    code(
        """
valid = folds[(folds.split == "valid") & (folds.base_model.isin(["v1", "lgbm"]))]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, base in zip(axes, ["v1", "lgbm"]):
    wide = valid[valid.base_model.eq(base)].pivot(index="fold", columns="estimator", values="brier")
    for estimator in ["amplitude_logit", "compact_logit"]:
        improvement = 1e4 * (wide["constant"] - wide[estimator])
        ax.plot(wide.index, improvement, marker="o", lw=2,
                color=COLORS[estimator], label=estimator.replace("_logit", ""))
    ax.axhline(0, color="black", lw=1)
    ax.set_title(base.upper())
    ax.set_xlabel("fold")
    ax.set_xticks(range(1, 9))
axes[0].set_ylabel("amélioration Brier × 10 000")
axes[1].legend()
fig.suptitle("Les probabilités d'erreur battent le constant dans les 8 folds", y=1.02)
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
## 2. Calibration : mieux discriminer ne signifie pas mieux calibrer

Le modèle amplitude suit très bien la diagonale. Le compact sépare davantage les cas
faciles, mais surestime l'erreur dans ses déciles les plus risqués. Son Brier global est
meilleur, tandis que son erreur de calibration par décile est moins bonne.
"""
    ),
    code(
        """
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
for ax, base in zip(axes, ["v1", "lgbm"]):
    for estimator in ["amplitude_logit", "compact_logit"]:
        d = calibration[(calibration.base_model.eq(base)) & (calibration.estimator.eq(estimator))]
        ax.plot(100*d.mean_predicted_error, 100*d.actual_error_rate, marker="o", lw=2,
                color=COLORS[estimator], label=estimator.replace("_logit", ""))
    lo, hi = 40, 53
    ax.plot([lo, hi], [lo, hi], color="black", ls="--", lw=1, label="calibration parfaite")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_title(base.upper())
    ax.set_xlabel("erreur prédite (%)")
axes[0].set_ylabel("erreur observée (%)")
axes[1].legend()
fig.suptitle("Calibration OOF par décile", y=1.02)
plt.tight_layout()
plt.show()

ece = (calibration.assign(abs_gap=lambda d:(d.mean_predicted_error-d.actual_error_rate).abs())
       .groupby(["base_model","estimator"])
       .apply(lambda d: np.average(d.abs_gap, weights=d.n_rows), include_groups=False)
       .rename("ECE").reset_index())
ece[ece.estimator.ne("constant")].round(5)
"""
    ),
    md(
        """
## 3. Utilité concrète : isoler les observations les plus fiables

Si l'on conserve les 10 % de lignes auxquelles le compact attribue la plus faible
probabilité d'erreur, l'accuracy atteint 58,36 % pour V1 et 58,32 % pour LightGBM.
L'abstention n'est pas permise par la métrique du challenge, mais ce test montre que
l'estimateur apprend bien une notion utile de difficulté.
"""
    ),
    code(
        """
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, base in zip(axes, ["v1", "lgbm"]):
    for estimator in ["amplitude_logit", "compact_logit"]:
        d = abstention[(abstention.base_model.eq(base)) & (abstention.estimator.eq(estimator))]
        ax.plot(100*d.coverage, 100*d.accuracy, marker="o", lw=2,
                color=COLORS[estimator], label=estimator.replace("_logit", ""))
    ax.set_title(base.upper())
    ax.set_xlabel("couverture, lignes les plus sûres (%)")
    ax.set_xticks(range(10, 101, 10))
axes[0].set_ylabel("accuracy (%)")
axes[1].legend()
fig.suptitle("Le compact identifie mieux l'extrême facile", y=1.02)
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
## 4. Pourquoi cela ne choisit-il pas correctement entre les modèles ?

La difficulté est largement commune à V1 et LightGBM. Sur les désaccords, le modèle
direct affiche une AUC de 0,5369 au train mais seulement 0,5053 en validation. Il apprend
des différences — notamment par allocation — qui se transportent mal vers les dates
tenues à l'écart.
"""
    ),
    code(
        """
auc = (folds[folds.estimator.isin(["amplitude_logit", "compact_logit", "direct_compact"])]
       .groupby(["base_model", "estimator", "split"]).roc_auc.mean().reset_index())
labels = []
train_values = []
valid_values = []
for base, estimator, label in [
    ("v1", "amplitude_logit", "V1 amplitude"),
    ("v1", "compact_logit", "V1 compact"),
    ("lgbm", "compact_logit", "LGBM compact"),
    ("selector", "direct_compact", "choix direct"),
]:
    d = auc[(auc.base_model.eq(base)) & (auc.estimator.eq(estimator))].set_index("split")
    labels.append(label); train_values.append(d.loc["train", "roc_auc"])
    valid_values.append(d.loc["valid", "roc_auc"])
x = np.arange(len(labels)); width=.36
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x-width/2, train_values, width, label="train", color="#A5A5A5")
ax.bar(x+width/2, valid_values, width, label="validation OOF", color="#5B9BD5")
ax.axhline(.5, color="black", ls="--", lw=1)
ax.set_xticks(x, labels)
ax.set_ylim(.49, .545)
ax.set_ylabel("ROC-AUC")
ax.set_title("Le choix relatif perd presque tout son signal hors date")
ax.legend()
plt.show()
"""
    ),
    md(
        """
## 5. Les gains des sélecteurs avec leurs barres d'incertitude

Les barres ci-dessous sont les quantiles 2,5–97,5 % du gain sur les 1 000 panels H5
appariés de 120 dates. Elles sont toutes beaucoup plus larges que leur moyenne.
"""
    ),
    code(
        """
plot_panels = panels[~panels.selector.isin(["V1", "LightGBM"])].copy()
order = ["amplitude_rank", "lower_error_amplitude", "lower_error_compact", "direct_compact"]
plot_panels = plot_panels.set_index("selector").loc[order].reset_index()
labels = ["rang amplitude", "erreur amplitude", "erreur compacte", "compact direct"]
mean = 100*plot_panels.mean_gain_vs_v1.to_numpy()
lower = 100*(plot_panels.mean_gain_vs_v1-plot_panels.q025_gain_vs_v1).to_numpy()
upper = 100*(plot_panels.q975_gain_vs_v1-plot_panels.mean_gain_vs_v1).to_numpy()
fig, ax = plt.subplots(figsize=(10, 5))
ax.errorbar(np.arange(4), mean, yerr=[lower, upper], fmt="o", capsize=6,
            color="#4472C4", markersize=8)
ax.axhline(0, color="black", ls="--", lw=1)
ax.set_xticks(np.arange(4), labels)
ax.set_ylabel("gain vs V1 (points d'accuracy)")
ax.set_title("Aucun gain n'est identifié avec une incertitude acceptable")
plt.show()

gate
"""
    ),
    code(
        """
fold_selector = selectors[selectors.fold.astype(str).ne("overall")].copy()
wide = fold_selector.pivot(index="fold", columns="selector", values="accuracy")
gain = 100*(wide[order].sub(wide["V1"], axis=0))
fig, ax = plt.subplots(figsize=(11, 5))
for selector, label in zip(order, labels):
    ax.plot(gain.index.astype(int), gain[selector], marker="o", lw=1.8, label=label)
ax.axhline(0, color="black", ls="--", lw=1)
ax.set_xticks(range(1, 9))
ax.set_xlabel("fold")
ax.set_ylabel("gain vs V1 (points)")
ax.set_title("Le signe du gain change selon les dates")
ax.legend(ncol=2)
plt.show()
"""
    ),
    md(
        """
## 6. Conclusion

- **Validé :** H6 améliore l'estimation de la difficulté absolue et identifie mieux les
  lignes très fiables.
- **À calibrer :** le compact discrimine mieux, mais ses probabilités sont moins bien
  calibrées que celles de l'amplitude seule.
- **Rejeté :** utiliser H6 pour choisir V1 ou LightGBM. Aucun sélecteur n'atteint le gate
  de 7/8 folds et 90 % de panels gagnants.

La prochaine étape ne doit pas être un modèle plus complexe. Il faut d'abord calibrer
strictement le compact pour l'usage « difficulté absolue ». Le choix relatif constitue
une autre cible et devra travailler sur la différence résiduelle d'erreurs, pas sur deux
probabilités de difficulté presque identiques.
"""
    ),
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUTPUT)
print(OUTPUT)

