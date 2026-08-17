"""Build the H13 paper-tuned Random Forest diagnostics notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "notebooks" / "H13_paper_tuned_random_forest.ipynb"


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
# H13 — Random Forest très régularisée, tunée selon la littérature

Objectif : vérifier proprement si des arbres très randomisés et fortement
contraints peuvent extraire les interactions allocation–marché que la V2
linéaire ne voit pas.

Le tuning est **imbriqué par dates**, sans target encoding. Le nombre final
d'arbres est fixé à 1 000 ; il n'est pas sélectionné sur les folds externes.
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
if (cwd/"gpt"/"outputs"/"h13_paper_tuned_random_forest").exists(): repo=cwd
elif (cwd.parent/"outputs"/"h13_paper_tuned_random_forest").exists(): repo=cwd.parent.parent
else: raise FileNotFoundError("Lancer depuis la racine ou gpt/notebooks.")
out=repo/"gpt"/"outputs"/"h13_paper_tuned_random_forest"
ref=repo/"gpt"/"outputs"/"h11_conditional_allocation_response"/"oof_baseline_raw.csv"
summary=json.loads((out/"summary.json").read_text())
inner=pd.read_csv(out/"inner_results.csv")
outer=pd.read_csv(out/"outer_results.csv")
curves=pd.read_csv(out/"tree_count_curves.csv")
structures=pd.read_csv(out/"structures.csv")
importance=pd.read_csv(out/"feature_importance.csv")
h13=pd.read_csv(out/"oof_predictions.csv",index_col="ROW_ID")
v2=pd.read_csv(ref,index_col="ROW_ID").loc[h13.index]
summary
"""
    ),
    md(
        """
## 1. Espace de régularisation et sélection imbriquée

Quatre régimes sont comparés dans le train de chaque fold externe. Ils font
varier ensemble `max_features`, la fraction de bootstrap, la taille minimale
des feuilles, la profondeur, le nombre maximal de feuilles et l'élagage.

La couleur indique la configuration finalement retenue par la validation
interne. Les différences sont petites : la sélection elle-même est bruitée,
d'où la nécessité du niveau externe.
"""
    ),
    code(
        """
order=["ultra_random","very_strong","strong","pruned"]
colors={"ultra_random":"#4472C4","very_strong":"#70AD47",
        "strong":"#ED7D31","pruned":"#C00000"}
fig,ax=plt.subplots(figsize=(11,5.6))
for name in order:
 d=inner[inner.config.eq(name)]
 ax.plot(d.outer_fold,100*d.valid_accuracy,marker="o",lw=1.8,
         color=colors[name],label=name)
for row in outer.itertuples():
 chosen=inner[(inner.outer_fold.eq(row.outer_fold)) &
              (inner.config.eq(row.selected_config))].iloc[0]
 ax.scatter(row.outer_fold,100*chosen.valid_accuracy,s=130,facecolors="none",
            edgecolors=colors[row.selected_config],linewidths=2.5)
ax.set_xticks(range(1,9));ax.set_xlabel("fold externe")
ax.set_ylabel("accuracy de validation interne (%)")
ax.set_title("La complexité optimale change selon les dates")
ax.legend(ncol=2);plt.tight_layout();plt.show()
outer[["outer_fold","selected_config","accuracy_gap"]]
"""
    ),
    md(
        """
## 2. Verdict OOF avec incertitude appariée par date

La forêt atteint **52,371 %**, contre **52,500 %** pour V2 : −0,129 point.
L'erreur standard par date vaut 0,110 point et l'IC95 du gain est
**[−0,344 ; +0,086] point**. Quatre folds sont gagnés, quatre perdus.

Le screening sur deux folds (+0,135 point) était donc un faux positif. C'est
précisément la raison de ne pas conclure à partir des deux premiers folds.
"""
    ),
    code(
        """
paired=pd.DataFrame({"fold":h13.fold.astype(int),"RF":h13.is_correct,"V2":v2.is_correct})
folds=paired.groupby("fold")[["RF","V2"]].mean()
folds["gain"]=folds.RF-folds.V2
fig,ax=plt.subplots(figsize=(10.5,5.2))
ax.bar(folds.index,100*folds.gain,
       color=np.where(folds.gain>0,"#70AD47","#C00000"))
ax.axhline(0,color="black",lw=1)
ax.axhline(100*summary["gain_vs_v2"],color="#4472C4",ls="--",lw=2,
           label=f"gain global = {100*summary['gain_vs_v2']:+.3f} pt")
ax.fill_between([0.5,8.5],100*summary["ci95_low"],100*summary["ci95_high"],
                color="#4472C4",alpha=.12,label="IC95 apparié par date")
ax.set_xlim(.5,8.5);ax.set_xlabel("fold externe")
ax.set_ylabel("gain RF − V2 (points de %)")
ax.set_title("La forêt gagne 4 folds sur 8 mais perd en moyenne")
ax.legend();plt.tight_layout();plt.show()
folds
"""
    ),
    md(
        """
## 3. 1 000 arbres suffisent : la courbe a convergé

Le nombre d'arbres est un budget de stabilisation, pas un hyperparamètre choisi
au meilleur point externe. Accuracy et AUC plafonnent vers 600 arbres. Ajouter
encore des arbres ne corrige pas le biais du modèle.
"""
    ),
    code(
        """
avg=curves.groupby("n_estimators").agg(
    valid_mean=("valid_accuracy","mean"),valid_std=("valid_accuracy","std"),
    train_mean=("train_accuracy","mean"),oob_mean=("oob_accuracy","mean"),
    auc_mean=("valid_auc","mean")).reset_index()
fig,axes=plt.subplots(1,2,figsize=(13,4.8))
axes[0].plot(avg.n_estimators,100*avg.train_mean,marker="o",label="train")
axes[0].plot(avg.n_estimators,100*avg.oob_mean,marker="o",label="OOB")
axes[0].plot(avg.n_estimators,100*avg.valid_mean,marker="o",label="validation externe")
axes[0].fill_between(avg.n_estimators,100*(avg.valid_mean-avg.valid_std),
                     100*(avg.valid_mean+avg.valid_std),alpha=.12)
axes[0].set_ylabel("accuracy moyenne (%)");axes[0].legend()
axes[1].plot(avg.n_estimators,avg.auc_mean,marker="o",color="#ED7D31")
axes[1].set_ylabel("AUC externe moyenne")
for ax in axes: ax.set_xlabel("nombre d'arbres")
fig.suptitle("La variance Monte-Carlo est pratiquement stabilisée à 600 arbres",y=1.02)
plt.tight_layout();plt.show()
avg
"""
    ),
    md(
        """
## 4. Avons-nous vraiment régularisé « comme un bourrin » ?

Oui. Les folds `ultra_random` utilisent environ six feuilles énormes ; les
folds `pruned` plafonnent à 24 feuilles et ont encore plus de 1 180 observations
au quantile 5 % des feuilles. Le gap train–validation moyen est seulement
0,415 point.

Cela ne ressemble pas à un surapprentissage incontrôlé. Les folds faibles
signalent plutôt du biais et un changement de relation entre dates.
"""
    ),
    code(
        """
fig,axes=plt.subplots(1,2,figsize=(13,4.8))
for name,d in structures.groupby("config"):
 axes[0].scatter(d.outer_fold,d.leaves_mean,s=90,label=name,color=colors[name])
 axes[1].scatter(d.outer_fold,d.leaf_samples_q05,s=90,label=name,color=colors[name])
axes[0].set_ylabel("nombre moyen de feuilles par arbre")
axes[1].set_ylabel("quantile 5 % de la taille des feuilles")
for ax in axes: ax.set_xlabel("fold externe");ax.set_xticks(range(1,9))
axes[0].legend();fig.suptitle("Des arbres réellement petits, avec de très grandes feuilles",y=1.02)
plt.tight_layout();plt.show()
structures
"""
    ),
    md(
        """
## 5. Le biais positif baisse, mais le signal aussi

La forêt prédit 58,47 % de positifs contre 60,60 % pour V2 et une cible OOF à
50,72 %. Cette correction apparente ne suffit pas : AUC et Brier sont également
moins bons. Ce n'est donc pas seulement un mauvais seuil à 0,5.
"""
    ),
    code(
        """
metrics=pd.DataFrame({
 "accuracy":[summary["rf_accuracy"],summary["baseline_v2_accuracy"]],
 "AUC":[summary["rf_auc"],summary["v2_auc"]],
 "Brier":[summary["rf_brier"],summary["v2_brier"]],
 "taux positif":[summary["rf_positive_prediction_rate"],
                  summary["v2_positive_prediction_rate"]]},index=["Random Forest","V2"])
fig,axes=plt.subplots(1,3,figsize=(14,4.6))
for ax,col in zip(axes,["accuracy","AUC","Brier"]):
 vals=metrics[col]
 bars=ax.bar(vals.index,vals,color=["#70AD47","#4472C4"])
 ax.bar_label(bars,labels=[f"{v:.5f}" for v in vals],padding=3)
 ax.set_title(col);ax.set_ylim(vals.min()-.0025,vals.max()+.0025)
fig.suptitle("La forêt perd aussi en ranking et en calibration",y=1.02)
plt.tight_layout();plt.show()
metrics
"""
    ),
    md(
        """
## 6. Importance descriptive des variables

Les résidus et profils d'allocation sont effectivement utilisés. Mais
l'importance par diminution d'impureté favorise les variables continues et
partage arbitrairement l'importance entre variables corrélées : ce graphe
explique le fit, il ne constitue pas une sélection de features.
"""
    ),
    code(
        """
imp=(importance.groupby("feature").impurity_importance
     .agg(["mean","std"]).sort_values("mean").tail(15))
fig,ax=plt.subplots(figsize=(10,6.3))
ax.barh(imp.index,imp["mean"],xerr=imp["std"],color="#5B9BD5",alpha=.9)
ax.set_xlabel("diminution moyenne d'impureté ± écart-type entre folds")
ax.set_title("RET_1, profils d'allocation et résidus portent l'essentiel des splits")
plt.tight_layout();plt.show()
imp.sort_values("mean",ascending=False)
"""
    ),
    md(
        """
## Conclusion

- Le tuning imbriqué et multi-axes empêche de confondre profondeur et
  régularisation.
- La convergence à 600–1 000 arbres est acquise.
- Le gap moyen de 0,415 point exclut l'hypothèse d'un simple overfit massif.
- La RF perd 0,129 point contre V2, avec un IC95 qui contient zéro.
- Elle ne sera ni soumise seule, ni empilée sans nouveau gate indépendant.

Références : Probst, Wright & Boulesteix (2019), Mentch & Zhou (2020), Probst &
Boulesteix (2018), Breiman (2001). Les liens sont conservés dans le protocole
`discoveries/39_h13_paper_tuned_random_forest_protocol.md`.
"""
    ),
]


OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUTPUT)
print(OUTPUT)
