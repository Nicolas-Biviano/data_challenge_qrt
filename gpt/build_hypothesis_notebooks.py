"""Construit trois notebooks autonomes pour tester H1, H2 et H3."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "gpt" / "notebooks"
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def base_notebook(title: str, introduction: str):
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    notebook["cells"] = [
        md(f"# {title}\n\n{introduction}"),
        code(
            """from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents] if (path / "gpt").exists())
OUT = ROOT / "gpt" / "outputs" / "hypotheses_h1_h2_h3"
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
BLUE, ORANGE, RED, GREY, GREEN = "#2563A6", "#D97706", "#C44536", "#8A94A6", "#21865A"
print(f"Résultats lus depuis : {OUT}")"""
        ),
    ]
    return notebook


def save(notebook, filename: str):
    path = NOTEBOOK_DIR / filename
    nbf.write(notebook, path)
    print(path)


# ---------------------------------------------------------------------------
# H1 — entraînement sur les seules dates test-like
# ---------------------------------------------------------------------------
h1 = base_notebook(
    "H1 — Entraîner uniquement sur les dates test-like",
    """**Hypothèse.** Les dates de test ont presque toujours 278 lignes. Les dates train
les plus proches en taille en ont 276. En retirant les autres dates de
l'entraînement, le modèle pourrait mieux correspondre au régime public.

**Protocole honnête.** Les folds restent des groupes de dates entières et sont
strictement identiques entre les variantes. La comparaison est donc appariée
par date. Aucun ordre temporel n'est supposé et aucun target encoding n'est
utilisé.""",
)
h1["cells"] += [
    md(
        """## 1. Comparaison principale

Les barres représentent des intervalles à 95 % obtenus par bootstrap des dates
entières. Ils mesurent l'incertitude liée au choix des dates, pas seulement le
nombre de lignes."""
    ),
    code(
        """results = pd.read_csv(OUT / "h1_training_regime.csv")
results["gain_50_pp"] = 100 * (results.accuracy - .5)
results["low_50_pp"] = 100 * (results.accuracy_ci_low - .5)
results["high_50_pp"] = 100 * (results.accuracy_ci_high - .5)

fig, ax = plt.subplots(figsize=(10.5, 5.2))
x = np.arange(len(results))
colors = [BLUE] + [ORANGE if "C=0.01" in name else GREY for name in results.model]
ax.bar(x, results.gain_50_pp, color=colors, width=.72)
ax.errorbar(
    x, results.gain_50_pp,
    yerr=np.vstack([results.gain_50_pp-results.low_50_pp, results.high_50_pp-results.gain_50_pp]),
    fmt="none", ecolor="#263238", elinewidth=1.2, capsize=4,
)
ax.axhline(0, color="#4B5563", ls="--", lw=1)
ax.set_xticks(x, results.model, rotation=24, ha="right")
ax.set(title="Entraîner seulement sur les dates complètes n'améliore pas la CV appariée",
       xlabel="", ylabel="Points d'accuracy au-dessus de 50 %")
plt.tight_layout(); plt.show()
results[["model", "accuracy", "accuracy_ci_low", "accuracy_ci_high", "gain_vs_v2"]]"""
    ),
    md(
        """## 2. Différence appariée face à la référence

Une comparaison absolue peut masquer la covariance entre deux modèles évalués
sur les mêmes dates. Ici, on bootstrap directement la différence d'accuracy par
date. Si l'intervalle traverse zéro, le gain n'est pas établi."""
    ),
    code(
        """delta = results.iloc[1:].copy()
delta["gain_pp"] = 100 * delta.gain_vs_v2
delta["low_pp"] = 100 * delta.gain_ci_low
delta["high_pp"] = 100 * delta.gain_ci_high

fig, ax = plt.subplots(figsize=(9.5, 4.8))
y = np.arange(len(delta))
ax.errorbar(delta.gain_pp, y,
            xerr=np.vstack([delta.gain_pp-delta.low_pp, delta.high_pp-delta.gain_pp]),
            fmt="o", color=ORANGE, ecolor="#263238", capsize=4, markersize=7)
ax.axvline(0, color=RED, ls="--", lw=1.2)
ax.set_yticks(y, delta.model)
ax.set(title="Gain apparié des entraînements test-like face à toutes les dates",
       xlabel="Différence d'accuracy (points)", ylabel="")
ax.invert_yaxis()
plt.tight_layout(); plt.show()
delta[["model", "gain_pp", "low_pp", "high_pp"]]"""
    ),
    md(
        """## Conclusion H1

La meilleure variante test-like est **C = 0,01**, mais elle perd environ
**0,062 point d'accuracy** face à l'entraînement sur toutes les dates. Son
intervalle de différence à 95 % contient zéro.

**H1 n'est donc pas soutenue.** Le régime de taille des dates est un bon signal
de décalage train/test, mais filtrer les données réduit la diversité et la
quantité d'apprentissage sans apporter de relation cible plus stable. On garde
toutes les dates pour entraîner le modèle final."""
    ),
]
save(h1, "H1_test_like_training.ipynb")


# ---------------------------------------------------------------------------
# H2 — AUC et seuils dynamiques
# ---------------------------------------------------------------------------
h2 = base_notebook(
    "H2 — AUC, seuils et calibration par date",
    """**Questions.** Le modèle sait-il correctement ordonner les observations même
si le seuil 0,5 est mal calibré ? Combien gagnerait un seuil parfait par date ?
Peut-on prédire ce seuil uniquement à partir de caractéristiques de marché ?

Toutes les méthodes déployables sont évaluées hors échantillon avec des folds
de dates entières. L'oracle est volontairement isolé : il utilise les vraies
étiquettes de la date évaluée et constitue uniquement une borne descriptive.""",
)
h2["cells"] += [
    md(
        """## 1. Le modèle contient un faible signal de classement

Une AUC de 0,5 correspond au hasard. L'AUC répond à une autre question que
l'accuracy : elle mesure si un positif reçoit en général un score supérieur à
un négatif, indépendamment d'un seuil particulier."""
    ),
    code(
        """fold_auc = pd.read_csv(OUT / "h2_auc_by_fold.csv")
summary = json.load(open(OUT / "summary.json"))["H2"]
mean_auc = fold_auc.auc.mean()
se_auc = fold_auc.auc.std(ddof=1) / np.sqrt(len(fold_auc))

fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.scatter(fold_auc.fold, fold_auc.auc, s=55, color=BLUE, label="AUC par fold")
ax.axhline(.5, color=RED, ls="--", lw=1.2, label="Hasard")
ax.axhline(mean_auc, color=ORANGE, lw=2, label=f"Moyenne = {mean_auc:.4f}")
ax.fill_between([.7, 8.3], mean_auc-1.96*se_auc, mean_auc+1.96*se_auc,
                color=ORANGE, alpha=.15, label="IC 95 % de la moyenne des folds")
ax.set(xlim=(.7, 8.3), xlabel="Fold", ylabel="AUC", title="AUC OOF stable mais modeste")
ax.legend(frameon=False, ncol=2)
plt.tight_layout(); plt.show()
pd.DataFrame({"AUC globale": [summary["global_auc"]], "AUC moyenne folds": [mean_auc],
              "IC bas approximatif": [mean_auc-1.96*se_auc], "IC haut approximatif": [mean_auc+1.96*se_auc]})"""
    ),
    md(
        """## 2. Faire varier le seuil global

La courbe montre simultanément l'accuracy et le taux de prédictions positives.
Le seuil 0,5 prédit environ 60 % de positifs, mais forcer artificiellement ce
taux vers 50 % ne maximise pas l'accuracy."""
    ),
    code(
        """curve = pd.read_csv(OUT / "h2_threshold_curve.csv")
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True)
for scope, color in [("Toutes dates", BLUE), ("Dates complètes", ORANGE)]:
    d = curve[curve.scope.eq(scope)]
    axes[0].plot(d.threshold, d.accuracy, color=color, lw=2, label=scope)
    axes[1].plot(d.threshold, d.positive_rate, color=color, lw=2, label=scope)
for ax in axes:
    ax.axvline(.5, color=RED, ls="--", lw=1.1)
axes[0].set(title="Accuracy selon le seuil", xlabel="Seuil", ylabel="Accuracy")
axes[1].set(title="Taux positif selon le seuil", xlabel="Seuil", ylabel="Prédictions positives")
axes[1].yaxis.set_major_formatter(lambda x, pos: f"{100*x:.0f} %")
axes[0].legend(frameon=False); axes[1].legend(frameon=False)
plt.tight_layout(); plt.show()
curve.loc[curve.groupby("scope").accuracy.idxmax(), ["scope", "threshold", "accuracy", "positive_rate"]]"""
    ),
    md(
        """## 3. L'oracle par date : une borne, pas un score réalisable

Pour chaque date, l'oracle essaie tous les seuils **après avoir vu ses vraies
cibles**, puis conserve le meilleur. Son score est donc optimiste pour deux
raisons : fuite directe des labels et sélection du maximum parmi de nombreux
seuils. Il quantifie néanmoins la valeur théorique d'une calibration parfaite."""
    ),
    code(
        """oracle = pd.read_csv(OUT / "h2_oracle_threshold_by_date.csv")
weighted_natural = np.average(oracle.natural_accuracy, weights=oracle.n_rows)
weighted_oracle = np.average(oracle.oracle_accuracy, weights=oracle.n_rows)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
sns.histplot(oracle.natural_accuracy, bins=35, color=BLUE, alpha=.45, label="Seuil 0,5", ax=axes[0])
sns.histplot(oracle.oracle_accuracy, bins=35, color=ORANGE, alpha=.40, label="Oracle date", ax=axes[0])
axes[0].set(title="Accuracy par date", xlabel="Accuracy", ylabel="Nombre de dates")
axes[0].legend(frameon=False)
axes[1].scatter(oracle.target_positive_rate, oracle.oracle_threshold,
                s=np.clip(oracle.n_rows/8, 8, 40), alpha=.25, color=ORANGE)
axes[1].axhline(.5, color=RED, ls="--", lw=1)
axes[1].set(title="L'oracle exploite fortement le taux positif réel",
            xlabel="Taux positif réel de la date — inconnu au test", ylabel="Seuil oracle")
plt.tight_layout(); plt.show()
pd.DataFrame({"méthode": ["Seuil 0,5", "Oracle par date (non déployable)"],
              "accuracy pondérée": [weighted_natural, weighted_oracle]})"""
    ),
    md(
        """## 4. Peut-on apprendre le seuil avec le marché ?

On prédit le seuil d'une date à partir de résumés de ses variables observables
(moyennes, dispersions, taux de valeurs positives/manquantes, taille de date).
Le seuil est appris uniquement sur les autres folds. Ridge et Random Forest
testent respectivement une relation régulière et une relation non linéaire."""
    ),
    code(
        """learned = pd.read_csv(OUT / "h2_learned_thresholds.csv")
rows = pd.read_csv(OUT / "h2_learned_threshold_rows.csv")

per_date = (rows.groupby(["method", "TS"], as_index=False)
            .agg(correct=("is_correct_dynamic", "sum"), n=("is_correct_dynamic", "size")))
rng = np.random.default_rng(42)
uncertainty_rows = []
for method, frame in per_date.groupby("method", sort=False):
    for scope, sample in [("Toutes dates", frame), ("Dates complètes", frame[frame.n.eq(276)])]:
        correct, n = sample.correct.to_numpy(), sample.n.to_numpy()
        boot = np.empty(1500)
        for b in range(len(boot)):
            idx = rng.integers(0, len(sample), len(sample))
            boot[b] = correct[idx].sum() / n[idx].sum()
        uncertainty_rows.append([method, scope, correct.sum()/n.sum(),
                                 *np.quantile(boot, [.025, .975])])
uncertainty = pd.DataFrame(uncertainty_rows,
                           columns=["method", "scope", "score", "low", "high"])

plot = learned.melt(id_vars="method", value_vars=["accuracy", "accuracy_complete_dates"],
                    var_name="scope", value_name="score")
plot["scope"] = plot.scope.map({"accuracy": "Toutes dates", "accuracy_complete_dates": "Dates complètes"})
fig, ax = plt.subplots(figsize=(10.5, 5))
x = np.arange(len(learned)); width = .36
for offset, scope, color in [(-width/2, "Toutes dates", BLUE), (width/2, "Dates complètes", ORANGE)]:
    values = plot[plot.scope.eq(scope)].set_index("method").loc[learned.method, "score"].to_numpy()
    ax.bar(x+offset, 100*(values-.5), width, color=color, label=scope)
    u = uncertainty[uncertainty.scope.eq(scope)].set_index("method").loc[learned.method]
    centers = 100 * (u.score.to_numpy() - .5)
    ax.errorbar(x+offset, centers,
                yerr=np.vstack([100*(u.score-u.low), 100*(u.high-u.score)]),
                fmt="none", ecolor="#263238", elinewidth=1, capsize=3)
ax.axhline(0, color="#4B5563", ls="--", lw=1)
ax.set_xticks(x, learned.method, rotation=20, ha="right")
ax.set(title="Les seuils appris réduisent l'accuracy", xlabel="",
       ylabel="Points d'accuracy au-dessus de 50 %")
ax.legend(frameon=False)
plt.tight_layout(); plt.show()
learned"""
    ),
    md(
        """## Conclusion H2

- L'**AUC globale vaut 0,5352** : il existe un signal de classement faible mais
  réel dans les scores.
- L'oracle par date monte artificiellement à **60,91 %**, mais il voit les labels
  de la date et subit un fort biais de sélection. Ce n'est pas un candidat de
  soumission.
- Les trois seuils appris hors échantillon font moins bien que **0,5**, globalement
  et sur les dates complètes.

**H2 n'est pas soutenue avec les features de marché actuelles.** Le taux de 60 %
de positifs prédit est surprenant, mais le corriger vers 50 % ne suffit pas : il
faut améliorer les scores eux-mêmes avant de recalibrer leur seuil."""
    ),
]
save(h2, "H2_auc_and_thresholds.ipynb")


# ---------------------------------------------------------------------------
# H3 — anatomie des lignes unanimement fausses ou justes
# ---------------------------------------------------------------------------
h3 = base_notebook(
    "H3 — Anatomie précise des lignes où tous les modèles se trompent",
    """Cette version de H3 travaille au niveau **ligne**, pas au niveau date.

Les cinq prédictions OOF sont alignées par `ROW_ID`. Une ligne est `tous faux`
si les cinq modèles prédisent le mauvais signe, `tous justes` s'ils prédisent
tous le bon signe, et `mixte` sinon.

Les comparaisons descriptives utilisent exclusivement les variables X. La
cible n'intervient que pour **stratifier le diagnostic** entre `y=0` et `y=1`.
Cette séparation est indispensable : sans elle, deux effets opposés peuvent
s'annuler et produire une conclusion trompeuse.""",
)
h3["cells"][1].source += '\nROW_OUT = ROOT / "gpt" / "outputs" / "h3_row_error_analysis"'
h3["cells"] += [
    md(
        """## 1. Population analysée et premier red flag

L'accord extrême concerne 71 % des lignes. Mais la composition en cible est
très différente : les lignes tous justes sont majoritairement positives alors
que les lignes tous faux sont majoritairement négatives. Une comparaison brute
confondrait donc difficulté et biais de signe."""
    ),
    code(
        """consensus = pd.read_csv(ROW_OUT / "h3_row_consensus.csv", index_col="ROW_ID")
error_summary = pd.read_csv(ROW_OUT / "h3_row_error_count_summary.csv")
summary = json.load(open(ROW_OUT / "h3_row_summary.json"))

extreme = pd.DataFrame({
    "groupe": ["Tous justes", "Tous faux", "Mixtes"],
    "lignes": [summary["n_all_right"], summary["n_all_wrong"], summary["n_mixed"]],
    "taux positif": [summary["target_positive_all_right"], summary["target_positive_all_wrong"],
                      consensus.loc[consensus.all_wrong.isna(), "target"].mean()],
})
fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
sns.barplot(data=extreme, x="groupe", y="lignes", hue="groupe", legend=False,
            palette=[GREEN, RED, GREY], ax=axes[0])
axes[0].set(title="71 % des lignes sont des accords extrêmes", xlabel="", ylabel="Nombre de lignes")
sns.barplot(data=extreme, x="groupe", y="taux positif", hue="groupe", legend=False,
            palette=[GREEN, RED, GREY], ax=axes[1])
axes[1].axhline(.5, color="#4B5563", ls="--", lw=1)
axes[1].set(title="La composition de cible change fortement", xlabel="", ylabel="Taux positif", ylim=(.30, .70))
axes[1].yaxis.set_major_formatter(lambda x, pos: f"{100*x:.0f} %")
plt.tight_layout(); plt.show()
extreme"""
    ),
    md(
        """## 2. Gradient de difficulté : de 0 à 5 modèles en erreur

`aligned_RET_1 = (2y-1) × RET_1` est positif lorsque `RET_1` pointe dans le sens
de la cible. C'est une variable **diagnostique non déployable**, puisqu'elle
utilise `y`. Elle permet de vérifier si les modèles échouent surtout lorsque le
rendement récent pointe dans la mauvaise direction."""
    ),
    code(
        """gradient = pd.read_csv(ROW_OUT / "h3_row_error_gradient.csv")
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
axes[0].bar(gradient.error_count, gradient.n, color=[GREEN, "#74A98D", GREY, GREY, "#D98B7F", RED])
axes[0].set(title="Nombre de modèles en erreur", xlabel="Erreurs parmi 5 modèles", ylabel="Nombre de lignes")
axes[1].plot(gradient.error_count, 1e4*gradient.aligned_RET_1_median, marker="o", lw=2, color=BLUE)
axes[1].axhline(0, color="#4B5563", ls="--", lw=1)
axes[1].set(title="RET_1 s'oppose progressivement à la cible", xlabel="Erreurs parmi 5 modèles",
            ylabel="Médiane de RET_1 aligné à y (bp)")
plt.tight_layout(); plt.show()
gradient"""
    ),
    code(
        """score_diag = pd.read_csv(ROW_OUT / "h3_row_v2_score_diagnostics.csv")
score_diag["dans ±2 points du seuil"] = 100 * score_diag.within_002_of_threshold
fig, ax = plt.subplots(figsize=(9.5, 4.6))
plot_score = score_diag.copy()
plot_score["segment"] = plot_score["group"] + ", y=" + plot_score.target.astype(str)
sns.barplot(data=plot_score, x="segment", y="dans ±2 points du seuil", hue="group",
            palette={"Tous justes": GREEN, "Tous faux": RED}, ax=ax)
ax.set(title="Une grande part des cas extrêmes reste proche du seuil V2",
       xlabel="", ylabel="Lignes avec |score − 0,5| ≤ 0,02 (%)")
ax.tick_params(axis="x", rotation=15)
ax.legend(frameon=False, title="")
plt.tight_layout(); plt.show()
score_diag"""
    ),
    md(
        """## 3. Comparaison complète des distributions

Pour chaque feature numérique, quatre mesures complémentaires sont calculées :

- **SMD** : écart de moyenne en unités d'écart-type ;
- **SMD robuste** : écart de médiane normalisé par l'IQR ;
- **KS** : plus grande différence entre les fonctions de répartition ;
- **PSI** et Wasserstein par quantiles : changement de forme/displacement.

Avec plus de 370 000 lignes extrêmes, les p-values seraient presque toujours
minuscules et peu informatives. On rapporte donc des tailles d'effet, puis des
IC à 95 % par bootstrap de dates pour les principales différences globales."""
    ),
    code(
        """metrics = pd.read_csv(ROW_OUT / "h3_row_distribution_metrics.csv")
conditional = metrics[metrics.scope.isin(["y=0", "y=1"])].copy()
feature_order = (conditional.assign(abs_smd=conditional.smd.abs())
                 .groupby("feature").abs_smd.max().nlargest(16).index)
heat = (metrics[metrics.feature.isin(feature_order)]
        .pivot(index="feature", columns="scope", values="smd")
        .reindex(feature_order)[["Tous", "y=0", "y=1"]])
fig, ax = plt.subplots(figsize=(8.8, 7.2))
sns.heatmap(heat, annot=True, fmt=".2f", center=0, cmap="vlag", vmin=-2.3, vmax=2.3,
            linewidths=.5, cbar_kws={"label":"SMD : tous faux − tous justes"}, ax=ax)
ax.set(title="Les grands effets s'inversent selon le vrai signe", xlabel="Condition", ylabel="")
plt.tight_layout(); plt.show()
metrics[metrics.scope.eq("Tous")].head(20)[
    ["feature", "family", "smd", "robust_smd", "ks", "psi", "missing_diff_pp"]
]"""
    ),
    code(
        """bootstrap = pd.read_csv(ROW_OUT / "h3_row_smd_bootstrap.csv").head(15)
plot_boot = bootstrap.sort_values("smd")
fig, ax = plt.subplots(figsize=(9.5, 5.8))
ypos = np.arange(len(plot_boot))
ax.errorbar(plot_boot.smd, ypos,
            xerr=np.vstack([plot_boot.smd-plot_boot.smd_ci_low,
                            plot_boot.smd_ci_high-plot_boot.smd]),
            fmt="o", color=BLUE, ecolor="#263238", capsize=3)
ax.axvline(0, color=RED, ls="--", lw=1)
ax.set_yticks(ypos, plot_boot.feature)
ax.set(title="Les différences globales sont statistiquement stables mais très petites",
       xlabel="SMD tous faux − tous justes, IC 95 % bootstrap des dates", ylabel="")
plt.tight_layout(); plt.show()
bootstrap"""
    ),
    md(
        """### Lecture essentielle

Globalement, le plus grand SMD numérique vaut seulement environ 0,06. Mais à
cible fixée, `RET_1` vaut environ **+1,55 écart-type pour y=0** et **−1,52 pour
y=1**, avec KS ≈ 0,67. Ce n'est pas une nouvelle feature magique : cela dit que
les modèles ont raison lorsque `RET_1` pointe vers la cible et tort lorsqu'il
pointe dans le sens opposé. Sans connaître `y`, les deux régimes s'annulent."""
    ),
    code(
        """quantiles = pd.read_csv(ROW_OUT / "h3_row_quantiles.csv")
ret1 = quantiles[quantiles.feature.eq("RET_1") & quantiles.scope.isin(["y=0", "y=1"])]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, scope in zip(axes, ["y=0", "y=1"]):
    for group, color in [("Tous justes", GREEN), ("Tous faux", RED)]:
        d = ret1[ret1.scope.eq(scope) & ret1.group.eq(group)]
        ax.plot(d["quantile"], 100*d["value"], marker="o", color=color, label=group)
    ax.axhline(0, color="#4B5563", ls="--", lw=1)
    ax.set(title=f"Distribution de RET_1 conditionnelle à {scope}", xlabel="Quantile", ylabel="RET_1 (%)")
    ax.legend(frameon=False)
plt.tight_layout(); plt.show()"""
    ),
    md(
        """## 4. `SIGNED_VOLUME_1` : 75 % de valeurs manquantes, effet conditionnel

L'absence de `SIGNED_VOLUME_1` est légèrement plus fréquente sur les lignes
tous faux globalement (+1,06 point). Mais l'effet change de signe : parmi les
vrais positifs, les erreurs ont environ 6,8 points de missingness en plus ;
parmi les vrais négatifs, elles en ont environ 5,1 points de moins. La moyenne
globale cache donc une interaction `missingness × signe futur`."""
    ),
    code(
        """missing = metrics[metrics.feature.str.startswith("missing_SIGNED_VOLUME_")].copy()
missing["lag"] = missing.feature.str.extract(r"(\\d+)$").astype(int)
missing["delta_missing_pp"] = 100 * (missing.mean_all_wrong - missing.mean_all_right)
fig, ax = plt.subplots(figsize=(10.5, 4.6))
for scope, color in [("Tous", GREY), ("y=0", BLUE), ("y=1", ORANGE)]:
    d = missing[missing.scope.eq(scope)].sort_values("lag")
    ax.plot(d.lag, d.delta_missing_pp, marker="o", color=color, label=scope)
ax.axhline(0, color="#4B5563", ls="--", lw=1)
ax.set(title="Différence de taux de valeurs manquantes : tous faux − tous justes",
       xlabel="Lag de SIGNED_VOLUME", ylabel="Différence (points)", xticks=range(1, 21))
ax.legend(frameon=False)
plt.tight_layout(); plt.show()
missing[missing.lag.eq(1)][["scope", "mean_all_wrong", "mean_all_right", "delta_missing_pp", "smd", "ks"]]"""
    ),
    md(
        """## 5. Allocations : le principal signal catégoriel est souvent une prédiction quasi constante

Certaines allocations reçoivent presque toujours le même signe des cinq
modèles. Par exemple, sur `ALLOCATION_152`, V2 prédit positif dans 99,84 % des
cas ; le taux positif réel vaut 65,89 %. L'accuracy élevée de cette allocation
vient donc surtout de son déséquilibre de classe, pas d'une discrimination
ligne par ligne.

Les tableaux d'association allocation/erreur sont des **diagnostics utilisant
les erreurs OOF**, jamais un encodage injecté dans le modèle."""
    ),
    code(
        """behavior = pd.read_csv(ROW_OUT / "h3_row_category_behavior.csv")
alloc = behavior[behavior.category_type.eq("ALLOCATION") & behavior.n.ge(500)].copy()
fig, ax = plt.subplots(figsize=(7.2, 6.2))
ax.scatter(alloc.target_positive_rate, alloc.prediction_positive_logistique_v2,
           s=np.clip(alloc.n/30, 15, 90), alpha=.45, color=BLUE)
ax.plot([0, 1], [0, 1], color=RED, ls="--", lw=1)
for name in ["ALLOCATION_152", "ALLOCATION_31", "ALLOCATION_111", "ALLOCATION_105", "ALLOCATION_18"]:
    d = alloc[alloc.category.eq(name)].iloc[0]
    ax.annotate(name.replace("ALLOCATION_", "A"), (d.target_positive_rate, d.prediction_positive_logistique_v2),
                xytext=(4, 4), textcoords="offset points", fontsize=8)
ax.set(title="Effets fixes très polarisés pour certaines allocations", xlabel="Taux positif réel OOF",
       ylabel="Taux positif prédit par V2", xlim=(-.03, 1.03), ylim=(-.03, 1.03))
plt.tight_layout(); plt.show()
alloc.loc[alloc.category.isin(["ALLOCATION_152", "ALLOCATION_31", "ALLOCATION_111", "ALLOCATION_105", "ALLOCATION_18"]),
          ["category", "n", "target_positive_rate", "prediction_positive_logistique_v2",
           "accuracy_logistique_v2", "mean_error_count", "extreme_share"]]"""
    ),
    code(
        """categorical = pd.read_csv(ROW_OUT / "h3_row_categorical_metrics.csv")
a = categorical[categorical.column.eq("ALLOCATION")]
pivot = a.pivot(index="category", columns="scope", values=["risk_diff_pp", "n_extreme"])
eligible = pivot[(pivot[("n_extreme", "y=0")] >= 300) & (pivot[("n_extreme", "y=1")] >= 300)].copy()
eligible["same_direction"] = (eligible[("risk_diff_pp", "y=0")] * eligible[("risk_diff_pp", "y=1")] > 0)
eligible["min_abs_conditional"] = eligible[[('risk_diff_pp','y=0'), ('risk_diff_pp','y=1')]].abs().min(axis=1)
stable = eligible[eligible.same_direction].sort_values("min_abs_conditional", ascending=False).head(12)
plot = stable[[('risk_diff_pp','y=0'), ('risk_diff_pp','y=1')]].copy()
plot.columns = ["y=0", "y=1"]
fig, ax = plt.subplots(figsize=(8.5, 5.4))
sns.heatmap(plot, annot=True, fmt=".1f", center=0, cmap="vlag", linewidths=.5,
            cbar_kws={"label":"Écart au risque moyen (points)"}, ax=ax)
ax.set(title=f"Seulement {int(eligible.same_direction.sum())}/{len(eligible)} allocations gardent le même sens selon y",
       xlabel="Cible", ylabel="")
plt.tight_layout(); plt.show()
pd.DataFrame({
    "risque global (pp)": stable[("risk_diff_pp", "Tous")],
    "risque y=0 (pp)": stable[("risk_diff_pp", "y=0")],
    "risque y=1 (pp)": stable[("risk_diff_pp", "y=1")],
    "effet conditionnel minimal (pp)": stable["min_abs_conditional"],
})"""
    ),
    md(
        """## 6. Peut-on reconnaître une future ligne tous faux avec X seulement ?

Quatre LightGBM fortement régularisés sont évalués en cinq folds, avec les dates
gardées entières. `ALLOCATION` et `GROUP` sont des catégories brutes ; aucun
target encoding n'est utilisé. Les barres d'erreur sont ±1,96 erreur standard
entre folds.

L'évaluation conditionnelle à `y` est uniquement diagnostique : la cible future
n'est évidemment pas disponible au moment de prédire."""
    ),
    code(
        """predictions = pd.read_csv(ROW_OUT / "h3_row_classifier_predictions.csv")
fold_rows = []
for (block, fold), d in predictions.groupby(["block", "fold"], observed=True):
    for scope, s in [("Tous", d), ("y=0", d[d.target.eq(0)]), ("y=1", d[d.target.eq(1)])]:
        fold_rows.append([block, int(fold), scope, roc_auc_score(s.all_wrong, s.probability_all_wrong)])
fold_auc = pd.DataFrame(fold_rows, columns=["block", "fold", "scope", "auc"])
stats = fold_auc.groupby(["block", "scope"], as_index=False).auc.agg(["mean", "std", "count"]).reset_index()
stats["se"] = stats["std"] / np.sqrt(stats["count"])

order = ["Returns", "Volumes", "Structure + catégories", "Toutes X enrichies"]
fig, ax = plt.subplots(figsize=(11, 5.2))
x = np.arange(len(order)); width = .25
for offset, scope, color in [(-width, "Tous", BLUE), (0, "y=0", GREY), (width, "y=1", ORANGE)]:
    d = stats[stats.scope.eq(scope)].set_index("block").loc[order]
    ax.bar(x+offset, d["mean"], width, color=color, label=scope)
    ax.errorbar(x+offset, d["mean"], yerr=1.96*d.se, fmt="none", ecolor="#263238", capsize=3)
ax.axhline(.5, color=RED, ls="--", lw=1)
ax.set_xticks(x, order, rotation=15, ha="right")
ax.set(title="Faible signal global, comportement opposé selon la cible", xlabel="", ylabel="AUC", ylim=(.43, .61))
ax.legend(frameon=False)
plt.tight_layout(); plt.show()
pd.read_csv(ROW_OUT / "h3_row_classifier_metrics.csv")"""
    ),
    code(
        """importance = pd.read_csv(ROW_OUT / "h3_row_classifier_importance.csv")
importance = importance[importance.block.eq("Toutes X enrichies")].head(18).sort_values("gain_importance")
fig, ax = plt.subplots(figsize=(9.5, 6))
ax.barh(importance.feature, importance.gain_importance, color=ORANGE)
ax.set(title="Le classifieur d'erreur s'appuie surtout sur ALLOCATION",
       xlabel="Importance LightGBM moyenne (gain)", ylabel="")
plt.tight_layout(); plt.show()
importance.sort_values("gain_importance", ascending=False)"""
    ),
    md(
        """## 7. Conclusion précise

1. **Il n'existe pas de signature numérique globale forte** des lignes tous
   faux : les SMD globaux restent sous environ 0,06. Les intervalles bootstrap
   peuvent exclure zéro grâce au très grand échantillon, mais les effets restent
   minuscules.
2. **La séparation spectaculaire de `RET_1` est conditionnelle à la cible.**
   Elle décrit exactement le conflit momentum/réalisation future, mais ne peut
   être utilisée directement puisque le signe de `y` est inconnu.
3. **La missingness de `SIGNED_VOLUME_1` interagit avec le signe** : +6,8 points
   sur les erreurs quand `y=1`, −5,1 points quand `y=0`. Son effet global de
   +1,1 point était trompeusement petit.
4. **Les allocations dominent le diagnostic**, souvent parce que les modèles
   produisent un signe presque constant. Seules 12 allocations sur 275 avec
   support suffisant gardent un effet de difficulté de même direction pour les
   deux classes.
5. Le meilleur détecteur X-only atteint seulement **AUC ≈ 0,524** globalement.
   Il monte à ≈0,580 sur `y=1` mais tombe à ≈0,466 sur `y=0` : ce n'est pas un
   détecteur universel d'erreurs.

Les nouvelles transformations et interactions non linéaires constituent une
hypothèse séparée, **H4**, afin de ne pas les confondre avec le diagnostic H3."""
    ),
]
save(h3, "H3_hard_vs_easy_dates.ipynb")


# ---------------------------------------------------------------------------
# H4 — nouvelles features non linéaires et interactions groupées
# ---------------------------------------------------------------------------
h4 = base_notebook(
    "H4 — Features non linéaires et interactions groupées",
    """**Hypothèse.** Une petite base de transformations saturantes, appliquée à des
variables standardisées relativement à leur date ou à `TS × GROUP`, peut
représenter des effets que les modèles linéaires actuels ratent sans recourir à
un arbre profond.

H4 est une hypothèse de modélisation distincte du diagnostic H3. Ce notebook
préenregistre les transformations et le protocole avant toute comparaison de
scores. Aucun résultat H4 n'est encore revendiqué.""",
)
h4["cells"][1].source += '''
H4_OUT = ROOT / "gpt" / "outputs" / "h4_nonlinear_group_features"
H4F_OUT = ROOT / "gpt" / "outputs" / "h4_nested_forward"
H4G_OUT = ROOT / "gpt" / "outputs" / "h4_nested_atomic_residual"'''
h4["cells"] += [
    md(
        """## 1. Biais inductif des transformations

Une transformation n'est pas neutre : elle impose une forme de relation.

| Transformation | Hypothèse imposée | Risque principal |
|---|---|---|
| `tanh(z/c)` | effet monotone, fort au centre puis saturé | choix opportuniste de l'échelle `c` |
| `arctan(z)` | saturation plus lente, queues encore informatives | redondance avec `tanh` |
| `sign(z)·log(1+|z|)` | effet monotone avec compression douce | encore sensible aux erreurs de standardisation |
| hinges | relation linéaire par morceaux | trop de nœuds = multiple testing |
| `arcsin(2r−1)` | importance accrue des rangs extrêmes | amplification du bruit dans les queues |
| `tan(z)` | oscillations et pôles périodiques | instabilité numérique et surapprentissage |

`tan` est exclu. `arcsin` reste un test de queue séparé, uniquement sur un rang
borné `r ∈ [0,1]`."""
    ),
    code(
        """z = np.linspace(-3, 3, 500)
curves = pd.DataFrame({
    "z": z,
    "linéaire": z,
    "tanh(z/0.5)": np.tanh(z/.5),
    "tanh(z)": np.tanh(z),
    "arctan(z)": np.arctan(z),
    "signed-log": np.sign(z)*np.log1p(np.abs(z)),
})
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for column, color in zip(curves.columns[1:], [GREY, RED, BLUE, ORANGE, GREEN]):
    axes[0].plot(curves.z, curves[column], lw=2, label=column, color=color)
axes[0].axhline(0, color="#4B5563", lw=1); axes[0].axvline(0, color="#4B5563", lw=1)
axes[0].set(title="Bases monotones : différents degrés de saturation", xlabel="z robuste", ylabel="f(z)")
axes[0].legend(frameon=False, ncol=2)

r = np.linspace(.001, .999, 500)
axes[1].plot(r, np.arcsin(2*r-1), color=ORANGE, lw=2, label="arcsin(2r−1)")
axes[1].plot(r, 2*r-1, color=GREY, lw=1.5, label="rang linéaire")
axes[1].axhline(0, color="#4B5563", lw=1)
axes[1].set(title="arcsin amplifie les rangs extrêmes", xlabel="Rang r", ylabel="Transformation")
axes[1].legend(frameon=False)
plt.tight_layout(); plt.show()"""
    ),
    md(
        """## 2. Standardisation groupée X-only

Pour une variable `x`, H4 construit deux coordonnées relatives :

`z_TS = (x − médiane_TS(x)) / (IQR_TS(x) + ε)`

`z_TS×GROUP = (x − médiane_TS×GROUP(x)) / (IQR_TS×GROUP(x) + ε)`

Les statistiques sont calculées uniquement avec X. Sur une date de validation,
elles utilisent les lignes X de cette même date, exactement comme sur une date
de test disponible en batch. Aucun label et aucun target encoding ne sont
utilisés.

Si l'IQR est nul ou le sous-groupe trop petit, la valeur groupée revient vers la
standardisation par date, puis vers la transformation globale apprise dans le
fold train."""
    ),
    md(
        """## 3. Bibliothèque H4 préenregistrée

Pour limiter le biais de sélection, les variables et formes sont fixées avant
l'expérience.

### Variables sources

- `RET_1` ;
- rang de `RET_1` dans `TS` et `TS × GROUP` ;
- rang du turnover dans `TS` et `TS × GROUP` ;
- indicateur de disponibilité de `SIGNED_VOLUME_1` ;
- part de volumes signés positifs observés.

### Formes autorisées

- `z`, `tanh(z/0.5)`, `tanh(z)`, `arctan(z)` ;
- `sign(z)·log(1+|z|)` ;
- hinges aux rangs 20 %, 50 % et 80 % ;
- `arcsin(2r−1)` uniquement dans le test de queues.

Les échelles supplémentaires, fonctions périodiques et expansions polynomiales
de degré élevé sont exclues."""
    ),
    md(
        """## 4. Interactions testées par étapes

1. **H4-A — base globale :** formes non linéaires sans interaction catégorielle.
2. **H4-B — GROUP :** `GROUP × f(z)` avec shrinkage renforcé sur les pentes.
3. **H4-C — ALLOCATION ciblée :** uniquement
   `ALLOCATION × f(RET_1)`, avec pénalisation plus forte que H4-B.
4. **H4-D — disponibilité SV1 :**
   `missing(SV1) × f(RET_1)` et `missing(SV1) × GROUP`.
5. **H4-E — queue :** `arcsin(2r−1)` ajouté seul au meilleur bloc précédent.

Cette progression permet d'attribuer un éventuel gain. Les blocs ne sont pas
combinés automatiquement."""
    ),
    md(
        """## 5. Modèle mathématique

Pour une base `φ(z) = [z, tanh(z/0.5), tanh(z), arctan(z), signed-log(z)]`, le
modèle H4-B prend la forme :

`logit P(yᵢ=1|Xᵢ) = β₀ + βᵀXᵢ + aᵀφ(zᵢ) + a_GROUP(i)ᵀφ(zᵢ)`

La pénalisation est hiérarchisée :

`λ_base ||β||² + λ_global ||a||² + λ_group Σg ||a_g||²`

avec `λ_group > λ_global`. Pour H4-C, la pénalisation allocation est encore
plus forte. Le modèle conserve donc une forme commune et n'autorise une pente
spécifique que si plusieurs dates la soutiennent."""
    ),
    md(
        """## 6. Protocole et gates de décision

- huit folds identiques, groupés par dates entières et sans ordre temporel ;
- imputation et paramètres globaux ajustés dans le fold train ;
- accuracy globale et accuracy sur dates complètes ;
- différence appariée face à V2, avec IC 95 % par bootstrap des dates ;
- taux positif, AUC et stabilité par classe rapportés systématiquement ;
- gain exigé sur les deux régimes, pas seulement sur la moyenne globale ;
- aucun nouveau bloc si son prédécesseur échoue.

Le script applique ce gate séquentiellement : si H4-A ne gagne pas sur les deux
régimes, H4-B à H4-E ne sont pas exécutées."""
    ),
    md(
        """## 7. Résultat de l'exécution

H4-A a été exécutée sur les huit folds. Les intervalles à 95 % sont obtenus par
bootstrap des dates entières et la différence face à V2 est appariée sur les
mêmes lignes."""
    ),
    code(
        """results = pd.read_csv(H4_OUT / "results.csv")
uncertainty = pd.read_csv(H4_OUT / "uncertainty.csv")
plot_rows = uncertainty[uncertainty.model.isin(["V2", "H4-A_global_nonlinear"])].copy()
plot_rows["label"] = plot_rows.model.map({"V2": "V2", "H4-A_global_nonlinear": "H4-A"})
plot_rows["gain_50_pp"] = 100 * (plot_rows.accuracy - .5)
plot_rows["low_50_pp"] = 100 * (plot_rows.accuracy_ci_low - .5)
plot_rows["high_50_pp"] = 100 * (plot_rows.accuracy_ci_high - .5)

fig, ax = plt.subplots(figsize=(9.5, 4.8))
x = np.arange(2); width = .36
for offset, scope, color in [(-width/2, "Toutes dates", BLUE),
                             (width/2, "Dates complètes (276)", ORANGE)]:
    d = plot_rows[plot_rows.scope.eq(scope)].set_index("label").loc[["V2", "H4-A"]]
    ax.bar(x+offset, d.gain_50_pp, width, color=color, label=scope)
    ax.errorbar(x+offset, d.gain_50_pp,
                yerr=np.vstack([d.gain_50_pp-d.low_50_pp, d.high_50_pp-d.gain_50_pp]),
                fmt="none", ecolor="#263238", capsize=4)
ax.set_xticks(x, ["V2", "H4-A"])
ax.set(title="H4-A ne dépasse pas V2 sur les deux régimes",
       ylabel="Points d'accuracy au-dessus de 50 %", xlabel="")
ax.legend(frameon=False)
plt.tight_layout(); plt.show()
uncertainty[uncertainty.model.eq("H4-A_global_nonlinear")][
    ["scope", "accuracy", "accuracy_ci_low", "accuracy_ci_high",
     "gain_vs_v2", "gain_ci_low", "gain_ci_high"]
]"""
    ),
    md(
        """## 8. Ce que H4-A change réellement

La légère hausse d'AUC indique que les scores ordonnent marginalement mieux les
lignes. Mais au seuil 0,5, H4-A déplace la calibration vers moins de positifs.
Elle corrige donc davantage de vrais négatifs tout en perdant presque autant de
vrais positifs."""
    ),
    code(
        """comparison = results[results.stage.isin(["V2", "A"])].copy()
comparison["modèle"] = comparison.stage.map({"V2": "V2", "A": "H4-A"})
class_plot = comparison.melt(id_vars="modèle", value_vars=["accuracy_y0", "accuracy_y1"],
                             var_name="classe", value_name="score_classe")
class_plot["classe"] = class_plot.classe.map({"accuracy_y0": "y=0", "accuracy_y1": "y=1"})
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
sns.barplot(data=class_plot, x="modèle", y="score_classe", hue="classe",
            palette=[BLUE, ORANGE], ax=axes[0])
axes[0].set(title="Le gain sur y=0 est payé sur y=1", xlabel="", ylabel="Accuracy par classe", ylim=(.38, .66))
rate_plot = comparison.melt(id_vars="modèle", value_vars=["prediction_positive_rate", "auc"],
                            var_name="mesure", value_name="valeur")
sns.barplot(data=rate_plot, x="modèle", y="valeur", hue="mesure",
            palette=[RED, GREEN], ax=axes[1])
axes[1].set(title="Moins de positifs, AUC presque inchangée", xlabel="", ylabel="Valeur", ylim=(.50, .62))
plt.tight_layout(); plt.show()
comparison[["modèle", "accuracy", "accuracy_complete_dates", "auc", "balanced_accuracy",
            "accuracy_y0", "accuracy_y1", "prediction_positive_rate"]]"""
    ),
    code(
        """reference = pd.read_csv(ROOT / "gpt" / "outputs" / "classification_full" / "oof_logistic_C_0.003.csv")
candidate = pd.read_csv(H4_OUT / "oof_H4-A_global_nonlinear.csv")
folds = pd.concat([
    reference.groupby("fold").is_correct.mean().rename("V2"),
    candidate.groupby("fold").is_correct.mean().rename("H4-A"),
], axis=1)
folds["différence_pp"] = 100 * (folds["H4-A"] - folds["V2"])
fig, ax = plt.subplots(figsize=(9, 4.2))
colors = [GREEN if value > 0 else RED for value in folds.différence_pp]
ax.bar(folds.index.astype(int), folds.différence_pp, color=colors)
ax.axhline(0, color="#4B5563", lw=1)
ax.set(title="Différence H4-A − V2 par fold", xlabel="Fold", ylabel="Différence d'accuracy (points)")
plt.tight_layout(); plt.show()
folds"""
    ),
    md(
        """## 9. Décision H4

H4-A obtient **0,524618** d'accuracy globale contre **0,525001** pour V2, et
**0,518056** sur les dates complètes contre **0,518438**. Les différences
appariées sont négatives dans les deux régimes et leurs intervalles contiennent
zéro.

Le gate préenregistré exigeait un gain simultané sur les deux régimes. Il n'est
pas passé. **H4-B, H4-C, H4-D et H4-E n'ont donc pas été exécutées.** Cette
décision évite d'utiliser les résultats intermédiaires pour choisir
opportunément une interaction parmi plusieurs centaines.

H4-A apporte toutefois un diagnostic : les formes non linéaires réduisent le
biais positif, mais ne transforment pas cette meilleure symétrie en gain
d'accuracy. V2 reste le modèle retenu."""
    ),
]
h4["cells"] += [
    md(
        """## 10. Pourquoi une CV imbriquée après H4-A ?

H4-A testait les 48 transformations simultanément. H4-F a ensuite fait une
forward stepwise **par groupes** dans trois folds internes, avant d'évaluer la
sélection sur un fold externe intact. Cela protège le score externe du biais de
sélection, mais certains groupes contenaient encore plusieurs copies corrélées.

H4-G corrige ce point : les 42 colonnes candidates sont comparées séparément
dans huit familles de redondance. Une seule transformation représente chaque
famille. Après chaque ajout, la logistique est réajustée : la feature suivante
doit donc apporter du signal conditionnellement aux précédentes. Le gate exige
un gain global, un gain sur les dates complètes et au moins deux folds internes
positifs sur trois pour les deux mesures."""
    ),
    code(
        """h4f_uncertainty = pd.read_csv(H4F_OUT / "uncertainty.csv")
h4g_uncertainty = pd.read_csv(H4G_OUT / "uncertainty.csv")
comparison_uncertainty = pd.concat([
    h4f_uncertainty[h4f_uncertainty.model.isin(["V2", "H4-F nested forward"])],
    h4g_uncertainty[h4g_uncertainty.model.eq("H4-G atomic residual")],
], ignore_index=True)
label_map = {
    "V2": "V2",
    "H4-F nested forward": "H4-F groupes",
    "H4-G atomic residual": "H4-G atomique",
}
comparison_uncertainty["label"] = comparison_uncertainty.model.map(label_map)
comparison_uncertainty["gain_50_pp"] = 100 * (comparison_uncertainty.accuracy - .5)
comparison_uncertainty["low_50_pp"] = 100 * (comparison_uncertainty.accuracy_ci_low - .5)
comparison_uncertainty["high_50_pp"] = 100 * (comparison_uncertainty.accuracy_ci_high - .5)

fig, ax = plt.subplots(figsize=(10.5, 5.1))
labels = ["V2", "H4-F groupes", "H4-G atomique"]
x = np.arange(3); width = .35
for offset, scope, color in [(-width/2, "Toutes dates", BLUE),
                             (width/2, "Dates complètes (276)", ORANGE)]:
    d = comparison_uncertainty[comparison_uncertainty.scope.eq(scope)].drop_duplicates("label")
    d = d.set_index("label").loc[labels]
    ax.bar(x + offset, d.gain_50_pp, width, color=color, label=scope)
    ax.errorbar(x + offset, d.gain_50_pp,
                yerr=np.vstack([d.gain_50_pp-d.low_50_pp,
                                d.high_50_pp-d.gain_50_pp]),
                fmt="none", ecolor="#263238", capsize=4)
ax.set_xticks(x, labels)
ax.set(title="La sélection imbriquée ne produit pas de gain établi",
       xlabel="", ylabel="Points d'accuracy au-dessus de 50 %")
ax.legend(frameon=False)
plt.tight_layout(); plt.show()
comparison_uncertainty[["label", "scope", "accuracy", "gain_vs_v2",
                        "gain_ci_low", "gain_ci_high"]]"""
    ),
    md(
        """## 11. Famille stable ou transformation stable ?

La fréquence d'une famille répond à « quelle information revient ? ». La
fréquence atomique répond à « savons-nous sous quelle forme l'utiliser ? ».
Une famille fréquente avec des représentants différents signale une relation
probable mais une forme mal identifiée."""
    ),
    code(
        """family_frequency = pd.read_csv(H4G_OUT / "family_selection_frequency.csv")
feature_frequency = pd.read_csv(H4G_OUT / "feature_selection_frequency.csv")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
sns.barplot(data=family_frequency, y="family", x="n_outer_folds_selected",
            color=BLUE, ax=axes[0])
axes[0].set(title="Familles retenues par la forward H4-G",
            xlabel="Nombre de folds externes sur 8", ylabel="")
top_features = feature_frequency.sort_values("n_outer_folds_selected").tail(12)
sns.barplot(data=top_features, y="feature", x="n_outer_folds_selected",
            color=ORANGE, ax=axes[1])
axes[1].set(title="Formes atomiques : beaucoup moins stables",
            xlabel="Nombre de folds externes sur 8", ylabel="")
for ax in axes:
    ax.set_xlim(0, 8)
plt.tight_layout(); plt.show()
family_frequency"""
    ),
    code(
        """representatives = pd.read_csv(H4G_OUT / "representative_trace.csv")
turnover_reps = representatives[
    representatives.family.eq("turnover_date_rank") & representatives.is_representative
].copy()
turnover_reps["stable_pp"] = 100 * turnover_reps.stable_objective
rep_matrix = turnover_reps.pivot(index="feature", columns="outer_fold", values="stable_pp")
fig, ax = plt.subplots(figsize=(10.5, 4.8))
sns.heatmap(rep_matrix, annot=True, fmt="+.3f", cmap="RdYlGn", center=0,
            linewidths=.7, cbar_kws={"label": "Objectif stable (points)"}, ax=ax)
ax.set(title="TURNOVER par date : la famille revient, la forme change",
       xlabel="Fold externe", ylabel="Représentant choisi")
plt.tight_layout(); plt.show()
turnover_reps[["outer_fold", "feature", "stable_pp", "gain_global", "gain_complete"]]"""
    ),
    md(
        """## 12. Le gain interne se transporte-t-il au fold externe ?

Le graphique suivant compare chaque procédure à V2 sur exactement les mêmes
lignes de chaque fold externe. Une bonne sélection devrait produire des gains
de même signe sur la majorité des folds, pas uniquement de bons scores
internes."""
    ),
    code(
        """reference = pd.read_csv(ROOT / "gpt" / "outputs" / "classification_full" / "oof_logistic_C_0.003.csv")
h4f_oof = pd.read_csv(H4F_OUT / "oof_nested_forward.csv")
h4g_oof = pd.read_csv(H4G_OUT / "oof_nested_atomic_residual.csv")
fold_comparison = pd.concat([
    reference.groupby("fold").is_correct.mean().rename("V2"),
    h4f_oof.groupby("fold").is_correct.mean().rename("H4-F"),
    h4g_oof.groupby("fold").is_correct.mean().rename("H4-G"),
], axis=1)
fold_comparison["H4-F − V2 (pp)"] = 100 * (fold_comparison["H4-F"] - fold_comparison.V2)
fold_comparison["H4-G − V2 (pp)"] = 100 * (fold_comparison["H4-G"] - fold_comparison.V2)
plot_fold = fold_comparison[["H4-F − V2 (pp)", "H4-G − V2 (pp)"]]
ax = plot_fold.plot(kind="bar", figsize=(11, 4.8), color=[BLUE, ORANGE], width=.75)
ax.axhline(0, color="#263238", lw=1)
ax.set(title="Gains externes par fold : instabilité persistante",
       xlabel="Fold externe", ylabel="Gain d'accuracy (points)")
ax.legend(frameon=False)
plt.xticks(rotation=0); plt.tight_layout(); plt.show()
fold_comparison"""
    ),
    md(
        """## 13. Verdict H4-F/H4-G

- **H4-F groupée :** 0,525233, soit +0,000231 face à V2 ; IC 95 % du gain
  `[-0,000670 ; 0,001127]`.
- **H4-G atomique/résiduelle :** 0,524686, soit −0,000315 ; IC 95 %
  `[-0,001108 ; 0,000423]`.
- Sur dates complètes, H4-G obtient 0,518056 contre 0,518438 pour V2.
- H4-G a pris 709 secondes, soit 11 min 49 s.

La sélection atomique répond bien au problème de redondance, mais elle révèle
que la meilleure forme varie trop entre folds. `TURNOVER` par date reste la
famille la plus intéressante (4 folds sur 8), tandis qu'aucune de ses formes
n'est retenue plus d'une fois. `sv1_available`, parfaitement atomique, revient
dans 3 folds.

Conclusion : le signal de ces nouvelles familles est trop faible pour payer la
variance de sélection. **V2 reste la référence.** Une future bibliothèque de
très grande dimension devra d'abord être dédupliquée par corrélation X-only,
puis passer par cette même sélection imbriquée."""
    ),
]
save(h4, "H4_nonlinear_group_features.ipynb")
