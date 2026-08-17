"""Build a notebook summarizing coding practices learned during the project."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "notebooks" / "coding_best_practices_learned.ipynb"


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
# Les meilleurs tips de code appris pendant le projet

Ce notebook résume les pratiques Python, pandas, scikit-learn et Jupyter qui ont
été les plus utiles pendant le challenge. Les exemples sont petits et
autonomes : ils peuvent être exécutés sans charger le dataset complet.

L'idée générale est simple : rendre les expériences **correctes,
reproductibles, inspectables et difficiles à casser**.
"""
    ),
    code(
        """
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 0
rng = np.random.default_rng(RANDOM_STATE)
print("Environnement prêt — seed =", RANDOM_STATE)
"""
    ),
    md(
        """
## 1. Un notebook doit pouvoir être relancé du début à la fin

Le message suivant n'est pas une erreur :

```text
The autoreload extension is already loaded.
To reload it, use: %reload_ext autoreload
```

`%load_ext autoreload` charge l'extension une première fois. Si elle est déjà
chargée, Jupyter affiche simplement cet avertissement. Une initialisation
idempotente peut être écrite ainsi :

```python
ip = get_ipython()
loaded = ip.extension_manager.loaded
if "IPython.extensions.autoreload" not in loaded:
    %load_ext autoreload
%autoreload 2
```

`%autoreload 2` recharge automatiquement les modules Python modifiés. C'est
pratique pendant le développement, mais cela ne remplace jamais un
**Restart Kernel and Run All** avant de considérer le notebook terminé.

Bon réflexe : toutes les constantes, imports et fonctions nécessaires doivent
apparaître avant leur première utilisation. Une cellule ne doit pas dépendre
d'un ancien état invisible du kernel.
"""
    ),
    md(
        """
## 2. `next` récupère le premier élément qui satisfait une condition

`next` consomme un itérateur ou un générateur et retourne son prochain élément.
Dans le projet, il est utile pour retrouver une configuration par son nom.
"""
    ),
    code(
        """
configs = [
    {"name": "strong", "alpha": 100},
    {"name": "very_strong", "alpha": 1000},
]

selected = next(config for config in configs if config["name"] == "very_strong")
selected
"""
    ),
    md(
        """
Sans résultat, `next(...)` lève `StopIteration`. Si l'absence est normale,
fournir une valeur par défaut :
"""
    ),
    code(
        """
missing = next(
    (config for config in configs if config["name"] == "unknown"),
    None,
)
assert missing is None
missing
"""
    ),
    md(
        """
## 3. `# noqa` parle au linter, pas à Python

`# noqa` signifie « ne signale pas cette ligne ». Cela ne change absolument pas
l'exécution du code.

Exemple du projet : nous ajoutons parfois la racine du dépôt à `sys.path` avant
d'importer un module local. Le linter considère alors que l'import n'est pas en
haut du fichier et déclenche `E402` :

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.dataloader import ChallengeDataLoader  # noqa: E402
```

Préférer un code conforme au linter quand c'est simple. Utiliser un code précis
comme `# noqa: E402` plutôt que `# noqa`, qui masque toutes les alertes de la
ligne.
"""
    ),
    md(
        """
## 4. L'index est une donnée : ne jamais le perdre silencieusement

L'erreur de soumission « Invalid Index » vient souvent d'un index réordonné,
réinitialisé ou écrit dans un format différent du fichier test.

La méthode la plus sûre consiste à partir du `sample_submission` et à remplacer
uniquement la colonne de prédiction.
"""
    ),
    code(
        """
test_index = pd.Index([101, 105, 108], name="ROW_ID")
sample_submission = pd.DataFrame({"target": [0, 0, 0]}, index=test_index)
predictions = np.array([1, 0, 1])

submission = sample_submission.copy()
submission.iloc[:, 0] = predictions

assert submission.index.equals(sample_submission.index)
assert len(submission) == len(predictions)
assert submission.index.is_unique
submission
"""
    ),
    md(
        """
Avant d'écrire le CSV, vérifier systématiquement : longueur, ordre, unicité,
nom de l'index et absence de valeur manquante. Éviter `reset_index()` sauf si le
format demandé l'exige explicitement.
"""
    ),
    md(
        """
## 5. `groupby().agg()` résume ; `groupby().transform()` reste aligné

`agg` produit une ligne par groupe. `transform` retourne une valeur pour chaque
ligne originale : c'est ce qu'il faut pour créer une feature transversale.
"""
    ),
    code(
        """
frame = pd.DataFrame({
    "TS": ["D1", "D1", "D1", "D2", "D2"],
    "RET_1": [0.2, -0.1, 0.4, -0.2, 0.1],
})

summary_by_date = frame.groupby("TS").RET_1.agg(["mean", "std"])
frame["date_mean"] = frame.groupby("TS").RET_1.transform("mean")

display(summary_by_date)
frame
"""
    ),
    md(
        r"""
## 6. Une moyenne leave-one-out évite qu'une ligne se recopie elle-même

Pour caractériser « ce que font les autres lignes de ma date », la moyenne
simple contient la valeur de la ligne courante. La moyenne leave-one-out la
retire explicitement :

\[
\bar{x}_{-i}=\frac{\sum_j x_j-x_i}{n-1}.
\]
"""
    ),
    code(
        """
def leave_one_out_mean(data: pd.DataFrame, group: str, column: str) -> pd.Series:
    values = data[column].astype(float)
    grouped = data.groupby(group)[column]
    sums = grouped.transform("sum")
    counts = grouped.transform("count")
    valid = values.notna().astype(int)
    return (sums - values.fillna(0.0)) / (counts - valid).replace(0, np.nan)

frame["other_rows_mean"] = leave_one_out_mean(frame, "TS", "RET_1")
frame
"""
    ),
    md(
        """
Cette précaution n'est pas du target leakage ici, car elle porte sur `X`. Elle
évite néanmoins une feature artificiellement trop proche de la valeur propre de
la ligne. Pour la cible `y`, toute agrégation doit rester strictement dans le
train du fold.
"""
    ),
    md(
        """
## 7. Beaucoup de NaN : séparer disponibilité et valeur

Imputer directement une variable manquante à 70 % mélange deux phénomènes :

1. la variable est-elle disponible ?
2. quelle est sa valeur lorsqu'elle est disponible ?

Une représentation en deux parties conserve les deux informations.
"""
    ),
    code(
        """
volume = pd.Series([np.nan, 1.2, np.nan, -0.5, 0.3], name="SIGNED_VOLUME_1")
two_part = pd.DataFrame({
    "sv1_observed": volume.notna().astype("int8"),
    "sv1_value": volume.fillna(volume.median()),
})
two_part
"""
    ),
    md(
        """
Le modèle peut alors apprendre un effet de disponibilité distinct de l'effet
de la valeur. L'imputation doit être ajustée dans le train du fold, jamais sur
train + validation + test.
"""
    ),
    md(
        """
## 8. Mettre tout preprocessing appris dans le fold

Le scaler, l'imputer et l'encodeur catégoriel apprennent des statistiques. Les
ajuster avant la CV transmet de l'information de validation au train.

`Pipeline` et `ColumnTransformer` rendent la séparation difficile à oublier.
"""
    ),
    code(
        """
toy = pd.DataFrame({
    "RET_1": [0.2, np.nan, -0.1, 0.3, -0.2, 0.1],
    "ALLOCATION": ["A", "A", "B", "B", "C", "C"],
})
target = np.array([1, 0, 0, 1, 0, 1])

preprocessing = ColumnTransformer([
    ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
     ["RET_1"]),
    ("category", OneHotEncoder(handle_unknown="ignore"), ["ALLOCATION"]),
])
model = make_pipeline(
    preprocessing,
    LogisticRegression(C=0.01, max_iter=1_000, random_state=RANDOM_STATE),
)
model.fit(toy, target)
model.predict_proba(toy)[:, 1]
"""
    ),
    md(
        """
`handle_unknown="ignore"` est important : une catégorie absente du train mais
présente en validation ne doit pas faire planter la transformation.
"""
    ),
    md(
        """
## 9. Conserver les matrices sparse quand les catégories sont nombreuses

Un one-hot d'allocation contient surtout des zéros. Le convertir en tableau
dense gaspille beaucoup de mémoire. `scipy.sparse.hstack` permet d'ajouter des
blocs sans densifier.
"""
    ),
    code(
        """
numeric_block = sparse.csr_matrix([[0.2], [-0.1], [0.4]])
category_block = sparse.eye(3, format="csr")
design = sparse.hstack([numeric_block, category_block], format="csr")

print("forme :", design.shape)
print("valeurs stockées :", design.nnz, "sur", design.shape[0] * design.shape[1])
"""
    ),
    md(
        """
## 10. Les noms de features doivent suivre exactement les colonnes

Dans scikit-learn récent, les transformeurs utilisent généralement
`get_feature_names_out()`. Une classe personnalisée n'a cette méthode que si
nous l'implémentons.

L'erreur rencontrée :

```text
AttributeError: 'FixedEffectDesign' object has no attribute 'get_feature_names'
```

n'indiquait pas un problème du modèle : le notebook appelait une méthode qui
n'existait pas encore dans notre classe.
"""
    ),
    code(
        """
class SmallDesign:
    def __init__(self):
        self.numeric_names = ["RET_1"]
        self.encoder = OneHotEncoder(handle_unknown="ignore").fit(
            pd.DataFrame({"ALLOCATION": ["A", "B"]})
        )

    def get_feature_names(self) -> list[str]:
        categorical = self.encoder.get_feature_names_out(["ALLOCATION"])
        return self.numeric_names + categorical.tolist()

SmallDesign().get_feature_names()
"""
    ),
    md(
        """
Toujours vérifier :

```python
assert X_transformed.shape[1] == len(feature_names)
```

Sinon le graphique d'importance associera les coefficients aux mauvaises
variables.
"""
    ),
    md(
        """
## 11. Splitter par groupe, pas par ligne

Ici, toutes les lignes d'une même date partagent un contexte. Une CV ligne par
ligne placerait ce contexte à la fois dans le train et la validation.

`GroupKFold` illustre la règle :
"""
    ),
    code(
        """
dates = np.array(["D1", "D1", "D2", "D2", "D3", "D3"])
X_toy = np.arange(6).reshape(-1, 1)
y_toy = np.array([0, 1, 0, 1, 0, 1])

rows = []
for fold, (train_idx, valid_idx) in enumerate(
    GroupKFold(n_splits=3).split(X_toy, y_toy, groups=dates), 1
):
    train_dates = set(dates[train_idx])
    valid_dates = set(dates[valid_idx])
    assert train_dates.isdisjoint(valid_dates)
    rows.append({"fold": fold, "train_dates": train_dates, "valid_dates": valid_dates})
pd.DataFrame(rows)
"""
    ),
    md(
        """
Dans le challenge, les labels des dates sont anonymisés et ne définissent pas
un ordre temporel fiable. Nous utilisons donc des folds groupés et mélangés par
date, pas une `TimeSeriesSplit` fondée sur le nom `DATE_xxxx`.
"""
    ),
    md(
        """
## 12. Comparer deux modèles de manière appariée

Comparer deux erreurs standard indépendantes gaspille l'information : les deux
modèles prédisent les mêmes lignes. On calcule d'abord leur différence de
succès, puis sa variation entre dates.
"""
    ),
    code(
        """
comparison = pd.DataFrame({
    "TS": ["D1"] * 4 + ["D2"] * 4 + ["D3"] * 4,
    "baseline_correct": [1,0,1,0, 1,1,0,0, 1,0,0,1],
    "challenger_correct": [1,1,1,0, 1,0,0,0, 1,1,0,1],
})
comparison["row_gain"] = (
    comparison.challenger_correct - comparison.baseline_correct
)
gain_by_date = comparison.groupby("TS").row_gain.mean()
gain = comparison.row_gain.mean()
standard_error = gain_by_date.std(ddof=1) / np.sqrt(len(gain_by_date))

pd.Series({
    "gain": gain,
    "standard_error_by_date": standard_error,
    "ci95_low": gain - 1.96 * standard_error,
    "ci95_high": gain + 1.96 * standard_error,
})
"""
    ),
    md(
        """
Un gain OOF sans barre d'erreur peut n'être qu'une fluctuation de quelques
dates. Pour les très petits gains, répéter également plusieurs seeds sans
choisir le seed le plus favorable.
"""
    ),
    md(
        """
## 13. Accuracy, AUC, Brier et taux positif répondent à des questions différentes

- **Accuracy** : le signe final au seuil choisi est-il correct ?
- **AUC** : les positifs sont-ils mieux classés que les négatifs, indépendamment
  du seuil ?
- **Brier/log-loss** : les probabilités sont-elles informatives et calibrées ?
- **Taux positif** : le modèle présente-t-il un biais de classe visible ?
- **Gap train–validation** : la capacité apprise se transfère-t-elle ?

Une AUC meilleure avec une accuracy inférieure signifie souvent que le ranking
progresse mais que le niveau ou le seuil reste mal localisé. Ne jamais ajuster
un seuil sur les mêmes OOF puis annoncer leur accuracy comme une validation
indépendante.
"""
    ),
    md(
        """
## 14. Configurations explicites plutôt que paramètres dispersés

Une `dataclass` ou un dictionnaire sérialisable rend l'expérience lisible et
permet d'enregistrer exactement ce qui a été exécuté.
"""
    ),
    code(
        """
@dataclass(frozen=True)
class ExperimentConfig:
    model: str
    alpha: float
    n_splits: int
    random_state: int

config = ExperimentConfig(
    model="ridge_fixed_effects",
    alpha=100.0,
    n_splits=8,
    random_state=0,
)
print(json.dumps(asdict(config), indent=2))
"""
    ),
    md(
        """
Écrire séparément :

- `summary.json` pour le verdict machine-readable ;
- `fold_metrics.csv` pour les diagnostics ;
- `oof_predictions.csv` pour les comparaisons futures ;
- un fichier Markdown pour l'hypothèse, le gate et la décision.

Pour un calcul long, sauvegarder après chaque fold et prévoir `--resume`.
"""
    ),
    md(
        """
## 15. Les gates protègent contre les rabbit holes

Une bonne procédure sépare :

1. **protocole écrit avant calcul** ;
2. **screening économique** ;
3. **confirmation sur tous les folds** ;
4. **multi-seed si le petit gain subsiste** ;
5. **soumission ou ensemble seulement après le gate final**.

H15 illustre l'intérêt de cette discipline : +0,074 point au screening, puis
−0,020 point lors de la confirmation. Sans gate, nous aurions probablement
optimisé davantage un faux positif.
"""
    ),
    md(
        """
## Checklist finale

Avant de considérer une expérience terminée :

- [ ] le notebook fonctionne après redémarrage du kernel ;
- [ ] les imports et chemins ne dépendent pas du répertoire courant implicite ;
- [ ] les transformations sont ajustées dans le train du fold ;
- [ ] les dates restent entières dans chaque split ;
- [ ] l'index OOF et l'index de soumission sont conservés ;
- [ ] les NaN ont un traitement explicite ;
- [ ] les matrices one-hot restent sparse ;
- [ ] le nombre de noms de features correspond au nombre de colonnes ;
- [ ] accuracy, AUC, calibration, taux positif et gap sont inspectés ;
- [ ] le gain est apparié et accompagné d'une incertitude par date ;
- [ ] aucun seed, seuil ou poids n'est choisi après observation du résultat ;
- [ ] les paramètres et la décision sont sauvegardés ;
- [ ] un résultat négatif est documenté au lieu d'être oublié.

La meilleure optimisation de code apprise dans ce projet n'est pas une astuce
de syntaxe : c'est de rendre une mauvaise conclusion difficile à produire.
"""
    ),
]


OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUTPUT)
print(OUTPUT)
