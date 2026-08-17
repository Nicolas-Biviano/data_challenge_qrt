"""Genere le notebook de presentation de la V2."""

from pathlib import Path

import nbformat as nbf


OUTPUT = Path(__file__).resolve().parent / "model_v2_presentation.ipynb"
nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
nb["cells"] = [
    nbf.v4.new_markdown_cell(
        """# Modèle GPT V2 — classification logistique sparse

Ce notebook présente le modèle final, sa validation par dates entières, sa
stabilité et les variables qui contribuent le plus. La V2 n'utilise **ni ordre
des dates**, **ni target encoding**, **ni reconstruction des fenêtres**."""
    ),
    nbf.v4.new_code_cell(
        """from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
ROOT = Path.cwd()
if not (ROOT / "gpt").exists():
    ROOT = ROOT.parent
V1 = ROOT / "gpt" / "outputs" / "v1_recheck"
V2 = ROOT / "gpt" / "outputs" / "v2"
OOF_V2 = ROOT / "gpt" / "outputs" / "classification_full" / "oof_logistic_C_0.003.csv"
print(f"Racine du projet: {ROOT}")"""
    ),
    nbf.v4.new_markdown_cell(
        """## 1. Résultat principal

La V1 prédit un rendement continu avec une Ridge. La V2 prédit directement la
probabilité d'un signe positif avec une logistique L2."""
    ),
    nbf.v4.new_code_cell(
        """with (V1 / "cv_summary.json").open() as f:
    v1_summary = json.load(f)
with (V2 / "summary.json").open() as f:
    v2_summary = json.load(f)

comparison = pd.DataFrame([
    {
        "modèle": "V1 Ridge",
        "accuracy": v1_summary["scores"]["ts"]["accuracy"],
        "erreur standard TS": v1_summary["scores"]["ts"]["standard_error"],
        "score pénalisé TS": v1_summary["scores"]["ts"]["score"],
    },
    {
        "modèle": "V2 Logistique",
        "accuracy": v2_summary["cv"]["accuracy"],
        "erreur standard TS": v2_summary["cv"]["ts_standard_error"],
        "score pénalisé TS": v2_summary["cv"]["ts_penalized_score"],
    },
])
comparison"""
    ),
    nbf.v4.new_markdown_cell(
        """## 2. Validation

Les 2 522 valeurs de `TS` sont réparties dans huit folds. Toutes les allocations
d'une date restent ensemble. Le numéro contenu dans `DATE_xxxx` n'est jamais
utilisé comme information temporelle."""
    ),
    nbf.v4.new_code_cell(
        """v1_folds = pd.read_csv(V1 / "fold_metrics.csv")[["fold", "accuracy"]].assign(modèle="V1 Ridge")
v2_folds = pd.read_csv(V2 / "fold_metrics.csv")[["fold", "accuracy"]].assign(modèle="V2 Logistique")
folds = pd.concat([v1_folds, v2_folds], ignore_index=True)

fig, ax = plt.subplots(figsize=(9, 4))
sns.lineplot(data=folds, x="fold", y="accuracy", hue="modèle", marker="o", ax=ax)
ax.axhline(0.5, color="black", lw=1, ls="--")
ax.set_title("Accuracy par fold de dates")
ax.set_ylabel("Accuracy")
plt.show()
folds.pivot(index="fold", columns="modèle", values="accuracy")"""
    ),
    nbf.v4.new_markdown_cell("## 3. Stabilité par date et allocation"),
    nbf.v4.new_code_cell(
        """oof = pd.read_csv(OOF_V2, index_col="ROW_ID")
stability_ts = oof.groupby("TS")["is_correct"].agg(["size", "mean"]).rename(columns={"size": "n", "mean": "accuracy"})
stability_allocation = oof.groupby("ALLOCATION")["is_correct"].agg(["size", "mean"]).rename(columns={"size": "n", "mean": "accuracy"})

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(stability_ts["accuracy"], bins=35, ax=axes[0])
axes[0].axvline(v2_summary["cv"]["accuracy"], color="crimson", ls="--")
axes[0].set_title("Distribution de l'accuracy par TS")
sns.histplot(stability_allocation["accuracy"], bins=30, ax=axes[1])
axes[1].axvline(v2_summary["cv"]["accuracy"], color="crimson", ls="--")
axes[1].set_title("Distribution par allocation")
plt.tight_layout()
plt.show()

pd.DataFrame({
    "niveau": ["TS", "ALLOCATION"],
    "nombre": [len(stability_ts), len(stability_allocation)],
    "accuracy moyenne non pondérée": [stability_ts.accuracy.mean(), stability_allocation.accuracy.mean()],
    "écart-type": [stability_ts.accuracy.std(), stability_allocation.accuracy.std()],
})"""
    ),
    nbf.v4.new_markdown_cell(
        """## 4. Importance des variables

L'importance est mesurée par l'écart-type de la contribution
`coefficient × feature`, ce qui est plus lisible qu'un coefficient brut pour
comparer variables continues et one-hot."""
    ),
    nbf.v4.new_code_cell(
        """importance = pd.read_csv(V2 / "feature_importance.csv")
top = importance.head(25).sort_values("contribution_std")
fig, ax = plt.subplots(figsize=(9, 8))
ax.barh(top["feature"], top["contribution_std"], color="#4C78A8")
ax.set_title("Top 25 — importance par contribution standardisée")
ax.set_xlabel("std(coefficient × feature)")
plt.tight_layout()
plt.show()
importance.head(15)"""
    ),
    nbf.v4.new_markdown_cell("## 5. Vérification de la soumission"),
    nbf.v4.new_code_cell(
        """submission = pd.read_csv(V2 / "submission.csv", index_col="ROW_ID")
checks = pd.Series({
    "nombre de lignes": len(submission),
    "valeurs manquantes": int(submission.isna().sum().sum()),
    "classes": sorted(submission.iloc[:, 0].unique().tolist()),
    "taux positif": submission.iloc[:, 0].mean(),
})
display(checks)
submission.head()"""
    ),
    nbf.v4.new_markdown_cell(
        """## 6. Formule et reproduction

La formulation complète est dans `gpt/model_v2_math.md`. Pour régénérer la
soumission depuis la racine du projet :

```bash
python gpt/model_v2.py
```

Le journal détaillé des expériences acceptées et rejetées se trouve dans
`gpt/RESEARCH_LOG.md`."""
    ),
]

nbf.write(nb, OUTPUT)
print(OUTPUT)

