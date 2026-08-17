# 02 - Decouverte incrementale des familles de features

## Question

Peut-on ameliorer la V1 en ajoutant des informations economiquement plausibles
sur les rendements, la liquidite et le regime transversal de chaque date?

## Protocole constant

Chaque experience conserve:

- la meme Ridge `alpha=100`;
- les effets fixes `ALLOCATION` et `GROUP`;
- la pente `ALLOCATION x RET_1`;
- les huit memes folds de `TS`;
- le meme seuil zero sur la cible continue predite.

Une seule famille de variables change a la fois. Les agregats par date et par
groupe sont calcules exclusivement avec `X`. Aucun target encoding n'est
utilise.

## Features testees

### Rendements

- les 20 retards bruts;
- moyennes sur 3, 5, 10 et 20 jours;
- volatilites sur 5, 10 et 20 jours;
- downside volatility;
- proportion de signes positifs;
- changements de signe et autocorrelation;
- rendement recent rapporte a la volatilite.

### Liquidite

- turnover median;
- moyenne, moyenne absolue et volatilite des volumes signes;
- proportion de volumes positifs;
- correlation rendement-volume;
- interactions `RET_1 x volume` et `RET_1 x turnover`.

### Regime transversal

Pour plusieurs variables, les statistiques suivantes sont calculees par `TS`,
puis par `TS x GROUP`:

- moyenne;
- ecart-type;
- z-score de l'allocation;
- rang percentile de l'allocation.

## Resultats

| Experience | Variables numeriques | Accuracy | Score penalise TS | Delta accuracy vs V1 |
|---|---:|---:|---:|---:|
| V1 | 8 | 0.524709 | 0.522975 | 0.000000 |
| 20 rendements | 20 | 0.523616 | 0.521898 | -0.001093 |
| Dynamique des rendements | 34 | 0.523315 | 0.521619 | -0.001394 |
| Liquidite resumee | 18 | 0.524550 | 0.522823 | -0.000159 |
| Liquidite complete | 38 | 0.524185 | 0.522457 | -0.000524 |
| Regime par date | 60 | 0.523356 | 0.521638 | -0.001353 |
| Regime date-groupe | 60 | 0.522632 | 0.520910 | -0.002078 |
| Combinaison complete | 120 | 0.520605 | 0.518878 | -0.004104 |

## Decouvertes

### 1. Les retards ne sont pas interchangeables

La selection parcimonieuse de la V1 bat l'utilisation naive des 20 retards.
Certains lags ajoutent donc principalement du bruit. Une selection ou une
regularisation uniforme ne suffit pas a les exploiter.

### 2. La linearite limite les features de regime

Les moyennes, rangs et dispersions transversales ne sont pas utiles sous forme
de termes lineaires additifs. Cela ne demontre pas que le regime de date est
inutile: son effet peut dependre d'interactions et de seuils qu'une Ridge ne
represente pas naturellement.

### 3. La liquidite est proche de la neutralite

La version resumee ne perd que 0.016 point d'accuracy, alors que les volumes
bruts en perdent 0.052. Les statistiques synthetiques sont donc preferables aux
20 valeurs brutes, mais elles n'ameliorent pas encore la V1 seules.

### 4. L'empilement aveugle est nettement nocif

Le modele combine perd 0.41 point. Une V2 ne doit pas etre une accumulation de
features. Les prochains tests doivent utiliser un modele non lineaire ou un
ensemble de predictions validees OOF.

## Decision

- conserver la selection de rendements de la V1 pour la branche lineaire;
- ne pas integrer les familles testees directement dans la Ridge finale;
- tester un modele non lineaire sur les variables de regime;
- mesurer la complementarite des erreurs OOF avant tout blend;
- conserver les resumes de liquidite comme candidats secondaires.

## Reproductibilite

Le code se trouve dans `gpt/feature_experiments.py`. Les sorties sont dans
`gpt/outputs/v2_experiments/`.
