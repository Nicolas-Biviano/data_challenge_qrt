"""Build the H11 conditional-allocation-response research notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "notebooks" / "H11_conditional_allocation_response.ipynb"


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
# H11 — apprendre comment chaque allocation réagit à l'état observable

Nous ne cherchons plus à prédire une étiquette de régime futur. Nous apprenons
une fonction de réaction conditionnelle :

\[
\operatorname{logit}P(y_{i,t+1}>0)
=a_i+\beta^\top x_{i,t}+\gamma_{g(i)}^\top m_t+\delta_i^\top m_t.
\]

(m_t) contient huit états calculés leave-one-out uniquement avec les returns
retardés de `X`. Aucun target encoding ni ordre des dates n'est utilisé.
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
if (cwd / "gpt" / "outputs" / "h11_conditional_allocation_response").exists():
    repo = cwd
elif (cwd.parent / "outputs" / "h11_conditional_allocation_response").exists():
    repo = cwd.parent.parent
else:
    raise FileNotFoundError("Lancer depuis la racine ou gpt/notebooks.")
out = repo / "gpt" / "outputs" / "h11_conditional_allocation_response"
atomic_out = repo / "gpt" / "outputs" / "h11_atomic_state_screen"
results = pd.read_csv(out / "results.csv")
uncertainty = pd.read_csv(out / "paired_uncertainty.csv")
folds = pd.read_csv(out / "fold_metrics.csv")
atomic = pd.read_csv(atomic_out / "results.csv")
atomic_folds = pd.read_csv(atomic_out / "fold_metrics.csv")
svd = pd.read_csv(out / "response_svd.csv")
group_response = pd.read_csv(out / "mean_response_by_group.csv", index_col=0)
candidate = json.loads((out / "candidate_summary.json").read_text())
candidate
"""
    ),
    md(
        """
## 1. Les huit états observables

- moyenne, dispersion et breadth de `RET_1` dans la date ;
- momentum court et volatilité récente moyens ;
- niveau, momentum et dispersion du `GROUP` relativement au marché.

Chaque moyenne exclut la ligne prédite. Les états sont connus au moment de la
prédiction puisqu'ils sont construits à partir des lags fournis.
"""
    ),
    md(
        """
## 2. Le niveau allocation est le seul niveau positif

Les intervalles représentent la différence de correction avec V2, agrégée par
date. Les états seuls et les réponses `GROUP` n'améliorent pas la baseline. La
réponse propre à l'allocation gagne 6 folds sur 8, mais son intervalle contient
encore zéro.
"""
    ),
    code(
        """
core_names = ["state_main", "group_response", "allocation_response_strong",
              "hierarchical_response", "style_low_rank_response"]
pretty = {"state_main":"états seuls", "group_response":"GROUP × états",
          "allocation_response_strong":"ALLOCATION × états",
          "hierarchical_response":"GROUP + ALLOCATION",
          "style_low_rank_response":"style continu × états"}
d = results.merge(uncertainty, on="experiment")
d = d[d.experiment.isin(core_names)].copy()
d["label"] = d.experiment.map(pretty)
d = d.sort_values("gain_accuracy")
lower = d.gain_accuracy-d.ci95_low; upper=d.ci95_high-d.gain_accuracy
fig, ax = plt.subplots(figsize=(10.5, 5.3))
colors = ["#70AD47" if gain > 0 else "#C00000" for gain in d.gain_accuracy]
ax.barh(d.label, 100*d.gain_accuracy, color=colors, alpha=.85)
ax.errorbar(100*d.gain_accuracy, d.label,
            xerr=np.vstack([100*lower,100*upper]), fmt="none",
            ecolor="black", capsize=4)
ax.axvline(0,color="black",lw=1)
ax.set_xlabel("gain vs V2 (points de %, IC 95 % apparié par date)")
ax.set_title("Seules les réponses propres aux allocations sont positives")
plt.tight_layout(); plt.show()
d[["label","accuracy","gain_accuracy","ci95_low","ci95_high","folds_won"]]
"""
    ),
    md(
        """
Le gain est **+0,032 point d'accuracy**, avec un IC95 de **[-0,065 ; +0,128]**
point. L'AUC progresse également de 0,535416 à 0,535727 : le résultat ne vient
donc pas uniquement du seuil à 0,5.
"""
    ),
    md(
        """
## 3. Diagnostic de régularisation

L'échelle multiplie les colonnes `ALLOCATION × état`. Sous pénalisation L2,
une petite échelle impose un shrinkage plus fort. `0,10` sous-ajuste ; `0,35`
n'améliore plus la validation mais fait monter le train. Nous conservons le
point préenregistré `0,20`, sans sélectionner rétrospectivement `0,35`.
"""
    ),
    code(
        """
scale_names = ["allocation_response_very_strong", "allocation_response_strong",
               "allocation_response_moderate"]
scale_label = {"allocation_response_very_strong":.10,
               "allocation_response_strong":.20,
               "allocation_response_moderate":.35}
sf = folds[folds.experiment.isin(scale_names)].groupby("experiment")[["train_accuracy","accuracy"]].mean()
sf["scale"] = sf.index.map(scale_label); sf=sf.sort_values("scale")
fig, ax = plt.subplots(figsize=(9.5, 5.1))
ax.plot(sf.scale, 100*sf.train_accuracy, marker="o", lw=2, label="train")
ax.plot(sf.scale, 100*sf.accuracy, marker="o", lw=2, label="validation")
for _, row in sf.iterrows():
    ax.annotate(f"{100*row.accuracy:.3f}", (row.scale,100*row.accuracy),
                xytext=(0,-16), textcoords="offset points", ha="center")
ax.set_xlabel("échelle du bloc ALLOCATION × état")
ax.set_ylabel("accuracy moyenne des folds (%)")
ax.set_title("Le gain plafonne tandis que l'overfit augmente")
ax.legend(); plt.tight_layout(); plt.show()
sf
"""
    ),
    md(
        """
## 4. Screening atomique : pas de déclencheur unique

Chaque état est ensuite testé seul. Cette table est exploratoire et ne doit pas
servir à annoncer le meilleur score après sélection. La dispersion et le
momentum court sont légèrement positifs ; aucun ne reproduit seul le gain du
bloc multivarié. La volatilité récente est la seule clairement défavorable.
"""
    ),
    code(
        """
atomic_pretty = {
 "market_ret1_mean":"moyenne RET_1", "market_ret1_dispersion":"dispersion RET_1",
 "market_ret1_breadth":"breadth RET_1", "market_short_momentum":"momentum court",
 "market_recent_volatility":"volatilité récente", "group_ret1_relative":"niveau groupe relatif",
 "group_momentum_relative":"momentum groupe relatif",
 "group_dispersion_relative":"dispersion groupe relative"}
a=atomic.sort_values("gain_accuracy").copy(); a["label"]=a.state.map(atomic_pretty)
lower=a.gain_accuracy-a.ci95_low; upper=a.ci95_high-a.gain_accuracy
fig,ax=plt.subplots(figsize=(10.5,5.8))
ax.barh(a.label,100*a.gain_accuracy,
        color=["#70AD47" if x>0 else "#A5A5A5" for x in a.gain_accuracy])
ax.errorbar(100*a.gain_accuracy,a.label,xerr=np.vstack([100*lower,100*upper]),
            fmt="none",ecolor="black",capsize=3)
ax.axvline(0,color="black",lw=1)
ax.set_xlabel("gain vs V2 (points de %, IC 95 %)")
ax.set_title("Les réponses utiles sont multivariées")
plt.tight_layout();plt.show()
a[["label","accuracy","gain_accuracy","folds_won"]]
"""
    ),
    md(
        """
## 5. Il existe une structure latente, mais nos styles naïfs ne la captent pas

Le modèle fondé sur huit profils X-only (volatilité, autocorrélation, exposition
au marché, turnover...) tombe à 52,4005 %. Pourtant, la matrice des réponses
apprises par allocation est structurée : trois axes expliquent 72,9 % de sa
variance. On ne peut pas réinjecter ces axes full-data sans fuite ; il faudrait
les apprendre dans chaque fold externe.
"""
    ),
    code(
        """
fig,axes=plt.subplots(1,2,figsize=(13,5.2))
axes[0].bar(svd.component,100*svd.explained_response_variance,color="#4472C4")
axes[0].plot(svd.component,100*svd.cumulative_explained_response_variance,
             color="#C00000",marker="o",label="cumul")
axes[0].axhline(100*svd.cumulative_explained_response_variance.iloc[2],
                color="black",ls="--",lw=1)
axes[0].set_xlabel("axe SVD");axes[0].set_ylabel("variance des réponses (%)")
axes[0].set_title("Trois axes expliquent 72,9 %")
axes[0].legend()
matrix=group_response.copy()
image=axes[1].imshow(matrix,cmap="RdBu_r",aspect="auto",
                     vmin=-np.abs(matrix.to_numpy()).max(),
                     vmax=np.abs(matrix.to_numpy()).max())
axes[1].set_yticks(range(len(matrix)),[f"GROUP {i}" for i in matrix.index])
axes[1].set_xticks(range(len(matrix.columns)),
                   [atomic_pretty.get(x,x) for x in matrix.columns],rotation=55,ha="right")
axes[1].set_title("Réponse moyenne apprise par GROUP")
fig.colorbar(image,ax=axes[1],label="pente logit effective")
plt.tight_layout();plt.show()
"""
    ),
    md(
        """
## Conclusion et candidat

H11 soutient faiblement mais concrètement l'hypothèse : **on ne prédit pas bien
le marché, mais l'identité de l'allocation aide à prédire sa réaction à une
configuration de marché observée**.

Le candidat `ALLOCATION × états`, échelle 0,20, a été ajusté sur tout le train.
Son index de soumission est exactement celui du test et son taux de positifs est
60,36 %. Il mérite un probe public, mais ne remplace pas V2 avant confirmation :
le gain OOF reste plus petit que son incertitude.
"""
    ),
    code(
        """
pd.DataFrame([candidate]).T.rename(columns={0:"valeur"})
"""
    ),
    md(
        """
## Reproduction

```bash
.venv/bin/python gpt/h11_conditional_allocation_response.py
.venv/bin/python gpt/h11_atomic_state_screen.py
.venv/bin/python gpt/build_h11_candidate.py
.venv/bin/python gpt/build_h11_notebook.py
.venv/bin/python -m jupyter nbconvert --to notebook --execute \
  --inplace gpt/notebooks/H11_conditional_allocation_response.ipynb
```
"""
    ),
]


OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUTPUT)
print(OUTPUT)
