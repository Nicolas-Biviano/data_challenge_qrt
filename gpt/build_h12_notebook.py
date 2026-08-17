"""Build the H12 multiseed and powerful-model diagnostics notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "notebooks" / "H12_multiseed_and_powerful_models.ipynb"


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
# H12 — changer les seeds et tester des modèles réellement plus puissants

Cette étape répond à deux questions : le petit gain H11 survit-il à plusieurs
découpages de dates, et une représentation plus puissante extrait-elle mieux les
réponses conditionnelles des allocations ?

Les modèles et hyperparamètres sont gelés avant comparaison. Aucun « meilleur
seed » n'est retenu.
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
cwd=Path.cwd().resolve()
if (cwd/"gpt"/"outputs"/"h11_multiseed_audit").exists(): repo=cwd
elif (cwd.parent/"outputs"/"h11_multiseed_audit").exists(): repo=cwd.parent.parent
else: raise FileNotFoundError("Lancer depuis la racine ou gpt/notebooks.")
multi=repo/"gpt"/"outputs"/"h11_multiseed_audit"
low=repo/"gpt"/"outputs"/"h12_low_rank_factorization"
lgb=repo/"gpt"/"outputs"/"h12_lgbm_conditional"
seeds=pd.read_csv(multi/"seed_results.csv")
multi_summary=json.loads((multi/"summary.json").read_text())
low_curves=pd.read_csv(low/"training_curves.csv")
low_summary=json.loads((low/"summary.json").read_text())
lgb_curves=pd.read_csv(lgb/"training_curves.csv")
lgb_results=pd.read_csv(lgb/"results.csv")
lgb_summary=json.loads((lgb/"summary.json").read_text())
multi_summary
"""
    ),
    md(
        """
## 1. H11 est directionnellement fréquent, mais extrêmement faible

Quatre seeds sur cinq sont positifs et 25 folds sur 40 sont gagnés. Le gain
moyen tombe cependant à **+0,011 point**, avec un intervalle qui contient
largement zéro. Les barres représentent les IC appariés par date de chaque
répétition ; la ligne rouge est la moyenne des cinq seeds.
"""
    ),
    code(
        """
fig,ax=plt.subplots(figsize=(9.5,5.2))
x=np.arange(len(seeds)); gain=100*seeds.gain_accuracy
lower=100*(seeds.gain_accuracy-seeds.ci95_low)
upper=100*(seeds.ci95_high-seeds.gain_accuracy)
ax.bar(x,gain,color=["#70AD47" if v>0 else "#C00000" for v in gain])
ax.errorbar(x,gain,yerr=np.vstack([lower,upper]),fmt="none",ecolor="black",capsize=4)
ax.axhline(100*multi_summary["mean_gain_across_seeds"],color="#C00000",lw=2,
           label=f"moyenne = {100*multi_summary['mean_gain_across_seeds']:+.3f} pt")
ax.axhline(0,color="black",lw=1)
ax.set_xticks(x,[f"seed {s}" for s in seeds.seed])
ax.set_ylabel("gain H11 vs V2 (points de %)")
ax.set_title("Le seed 0 surestimait légèrement le gain moyen")
ax.legend();plt.tight_layout();plt.show()
seeds
"""
    ),
    md(
        r"""
## 2. Factorization machine low-rank

Le correcteur apprend une interaction bilinéaire de rang 4 :

\[
z_{i,t}=z^{V2}_{i,t}+w^\top m_t+u_i^\top V m_t.
\]

Il généralise moins bien dès le premier epoch. Après 12 epochs, il perd 0,57
point sur les deux folds et présente 1,63 point d'écart train–validation. Les
huit folds et les architectures plus profondes sont donc bloqués.
"""
    ),
    code(
        """
fig,axes=plt.subplots(1,2,figsize=(13,4.8),sharey=True)
for fold,ax in zip(sorted(low_curves.fold.unique()),axes):
 d=low_curves[low_curves.fold.eq(fold)]
 ax.plot(d.epoch,100*d.train_accuracy,marker="o",label="train")
 ax.plot(d.epoch,100*d.valid_accuracy,marker="o",label="validation")
 ax.axhline(100*low_summary["base_accuracy"],color="black",ls="--",label="V2 agrégée")
 ax.set_title(f"fold {fold}");ax.set_xlabel("epoch")
axes[0].set_ylabel("accuracy (%)");axes[1].legend()
fig.suptitle("Le modèle low-rank mémorise rapidement les allocations",y=1.02)
plt.tight_layout();plt.show()
pd.DataFrame([low_summary]).T.rename(columns={0:"valeur"})
"""
    ),
    md(
        """
## 3. LightGBM régularisé : profondeur fiable, généralisation insuffisante

Les arbres ont une profondeur réelle maximale de 3, de grandes feuilles et de
fortes pénalités. Les feuilles linéaires améliorent légèrement les constantes,
mais les deux restent sous V2. La capacité supplémentaire augmente surtout le
score train.
"""
    ),
    code(
        """
avg=lgb_curves.groupby(["leaf_type","iteration"])[["train_accuracy","valid_accuracy"]].mean().reset_index()
fig,axes=plt.subplots(1,2,figsize=(13,4.8),sharey=True)
for ax,leaf in zip(axes,["constant","linear"]):
 d=avg[avg.leaf_type.eq(leaf)]
 ax.plot(d.iteration,100*d.train_accuracy,marker="o",label="train")
 ax.plot(d.iteration,100*d.valid_accuracy,marker="o",label="validation")
 ax.axhline(100*lgb_summary["baseline_v2_same_folds"],color="black",ls="--",label="V2")
 ax.set_title(f"feuilles {leaf}");ax.set_xlabel("nombre d'arbres")
axes[0].set_ylabel("accuracy moyenne (%)");axes[1].legend()
fig.suptitle("Plus d'arbres élargit surtout le gap train–validation",y=1.02)
plt.tight_layout();plt.show()
lgb_results
"""
    ),
    md(
        """
## 4. Comparaison finale et décision

Les gains H11 sont minuscules mais plus résistants que ceux des modèles plus
flexibles. La puissance brute ne compense pas l'absence d'une meilleure
invariance ou d'une nouvelle information.
"""
    ),
    code(
        """
labels=["V2","H11 moyen 5 seeds","LightGBM constant","LightGBM linéaire","Low-rank (2 folds)"]
values=[multi_summary["mean_v2_accuracy"],multi_summary["mean_h11_accuracy"],
        lgb_results.set_index("leaf_type").loc["constant","accuracy"],
        lgb_results.set_index("leaf_type").loc["linear","accuracy"],
        low_summary["low_rank_accuracy"]]
fig,ax=plt.subplots(figsize=(10.5,5.2))
bars=ax.barh(labels,100*np.array(values),color=["#4472C4","#70AD47","#A5A5A5","#ED7D31","#C00000"])
ax.set_xlim(51.7,52.65);ax.set_xlabel("accuracy OOF (%)")
ax.set_title("Les modèles puissants ne dépassent pas la logistique régularisée")
ax.bar_label(bars,labels=[f"{100*v:.3f}" for v in values],padding=3)
plt.tight_layout();plt.show()
"""
    ),
    md(
        """
## Conclusion

- H11 passe un gate directionnel faible : 4/5 seeds positifs, 25/40 folds.
- Son gain moyen n'est que +0,011 point et reste non significatif.
- La factorization machine low-rank sur-apprend immédiatement.
- LightGBM classique et linéaire restent sous V2 malgré une profondeur 3
  vérifiée et une forte régularisation.
- Le gate interdit le stacking avec ces modèles.

La prochaine hypothèse devra ajouter une structure nouvelle — par exemple une
relation temporelle ou transversale plus pertinente — plutôt que simplement
augmenter la capacité.
"""
    ),
]


OUTPUT.parent.mkdir(parents=True,exist_ok=True)
nbf.write(nb,OUTPUT)
print(OUTPUT)
