"""Build the H14-H16 gated-resumption notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "notebooks" / "H14_H16_controlled_resumption.ipynb"


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
# H14–H16 — Reprise contrôlée et gates de validation

Ce notebook répond à trois questions pré-enregistrées :

1. une cible continue dé-factorisée est-elle plus transférable ?
2. une hiérarchie `GROUP → ALLOCATION` à seulement trois états améliore-t-elle V2 ?
3. le gain survit-il à un niveau de validation plus strict avant tout ensemble ?

Aucun seuil, seed ou poids de blend n'est choisi après lecture des résultats.
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
plt.rcParams.update({"figure.figsize":(10,5.2),"axes.titlesize":14,
                     "axes.labelsize":11,"legend.frameon":False})
cwd=Path.cwd().resolve()
if (cwd/"gpt"/"outputs"/"h14_cross_sectional_target").exists(): repo=cwd
elif (cwd.parent/"outputs"/"h14_cross_sectional_target").exists(): repo=cwd.parent.parent
else: raise FileNotFoundError("Lancer depuis la racine ou gpt/notebooks.")
h14=repo/"gpt"/"outputs"/"h14_cross_sectional_target"
h15=repo/"gpt"/"outputs"/"h15_low_dimensional_hierarchy"
h15c=h15/"confirm8"
t14=pd.read_csv(h14/"comparisons.csv")
f14=pd.read_csv(h14/"fold_results.csv")
s15=pd.read_csv(h15/"results.csv")
f15=pd.read_csv(h15/"fold_metrics.csv")
c15=pd.read_csv(h15c/"results.csv")
fc15=pd.read_csv(h15c/"fold_metrics.csv")
json.loads((h15c/"summary.json").read_text())
"""
    ),
    md(
        """
## H14 — Dé-factoriser la cible améliore un peu le ranking, pas le signe

Toutes les cibles transformées perdent en accuracy. `market_100` est la moins
mauvaise à −0,088 point et perd les quatre folds. Certaines variantes gagnent
un peu d'AUC : elles ordonnent mieux les rendements relatifs mais localisent
moins bien le zéro nécessaire à l'accuracy.
"""
    ),
    code(
        """
d=t14.set_index("target_variant").copy()
d["gain_auc"]=d.auc-d.loc["raw","auc"]
d=d.drop(index="raw").sort_values("gain_accuracy")
fig,axes=plt.subplots(1,2,figsize=(13,5.2))
axes[0].barh(d.index,100*d.gain_accuracy,
             color=np.where(d.gain_accuracy>0,"#70AD47","#C00000"))
axes[0].axvline(0,color="black",lw=1);axes[0].set_xlabel("gain d'accuracy (points de %)")
axes[1].barh(d.index,100*d.gain_auc,
             color=np.where(d.gain_auc>0,"#70AD47","#C00000"))
axes[1].axvline(0,color="black",lw=1);axes[1].set_xlabel("gain d'AUC (points)")
fig.suptitle("H14 : le ranking relatif progresse parfois, l'accuracy jamais",y=1.02)
plt.tight_layout();plt.show()
t14.sort_values("accuracy",ascending=False)
"""
    ),
    md(
        """
## H14 — Le retrait du niveau recentre brutalement les prédictions

Le taux positif passe de 54,4 % à environ 49 % pour les cibles complètement
centrées. Ce recentrage n'est pas une preuve d'amélioration : le meilleur taux
de classe ne compense pas la perte d'information sur le niveau futur.
"""
    ),
    code(
        """
order=t14.sort_values("accuracy",ascending=False).target_variant
fig,ax=plt.subplots(figsize=(10.5,5.2))
x=np.arange(len(order)); vals=t14.set_index("target_variant").loc[order]
ax.bar(x,100*vals.positive_prediction_rate,color="#5B9BD5")
ax.set_xticks(x,order,rotation=35,ha="right")
ax.set_ylabel("prédictions positives (%)")
ax.set_title("Une distribution plus équilibrée ne suffit pas à gagner")
plt.tight_layout();plt.show()
"""
    ),
    md(
        """
## H15 — Le screening 4-fold était encourageant

Réduire les huit états H11 à deux ou trois états aide. Les quatre challengers
gagnent contre V2 sur ce screening ; `hierarchical_strong` atteint +0,074 point
avec un gap train–validation de 0,53 point.
"""
    ),
    code(
        """
base=float(s15.loc[s15.experiment.eq("baseline_raw"),"accuracy"].iloc[0])
screen=s15[s15.experiment.ne("baseline_raw")].copy()
screen["gain"]=screen.accuracy-base
screen=screen.sort_values("gain")
fig,ax=plt.subplots(figsize=(10,5))
ax.barh(screen.experiment,100*screen.gain,
        color=np.where(screen.gain>0,"#70AD47","#C00000"))
ax.axvline(0,color="black",lw=1);ax.set_xlabel("gain vs V2 (points de %)")
ax.set_title("H15 screening : la basse dimension semble utile")
plt.tight_layout();plt.show()
screen[["experiment","accuracy","gain"]]
"""
    ),
    md(
        """
## H15 — La confirmation 8-fold renverse la décision

La spécification est gelée avant confirmation. Elle tombe de +0,074 à −0,020
point et ne gagne que trois folds sur huit. L'IC95 apparié du gain est
`[-0,115 ; +0,076]` point.
"""
    ),
    code(
        """
p=fc15.pivot(index="fold",columns="experiment",values="accuracy")
p["gain"]=p.hierarchical_strong-p.baseline_raw
fig,axes=plt.subplots(1,2,figsize=(13,4.8))
axes[0].bar(["screening 4-fold","confirmation 8-fold"],[0.0738038,-0.0195419],
            color=["#70AD47","#C00000"])
axes[0].axhline(0,color="black",lw=1);axes[0].set_ylabel("gain (points de %)")
axes[1].bar(p.index,100*p.gain,color=np.where(p.gain>0,"#70AD47","#C00000"))
axes[1].axhline(0,color="black",lw=1);axes[1].set_xlabel("fold externe")
axes[1].set_ylabel("gain H15 − V2 (points de %)")
fig.suptitle("Le niveau de validation supplémentaire bloque un faux positif",y=1.02)
plt.tight_layout();plt.show()
p
"""
    ),
    md(
        """
## H16 — Décision finale

| Étape | Décision |
|---|---|
| H14 screening | rejet, aucune variante ne gagne un fold |
| H15 screening | passage à la confirmation |
| H15 confirmation | rejet, gain négatif et 3/8 folds |
| Multi-seed | non autorisé |
| Panels test-sized | non autorisés |
| Ensemble ou soumission | non autorisé |

Le résultat le plus important est méthodologique : nous avons effectivement
arrêté un candidat qui paraissait bon au premier niveau. V2 reste la baseline
locale. La phase GitHub reste séparée et n'a fait l'objet d'aucune modification.
"""
    ),
]


OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUTPUT)
print(OUTPUT)
