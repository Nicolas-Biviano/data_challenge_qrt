# Wrap-up simple du travail de modélisation

## 1. La procédure utilisée

Nous avons travaillé de manière incrémentale : une hypothèse à la fois, puis une
comparaison avec une baseline fixe.

La validation respecte toujours les règles suivantes :

- les folds sont construits par date entière : une date ne peut pas être à la
  fois dans le train et dans la validation ;
- les dates anonymisées ne sont jamais interprétées comme un ordre chronologique ;
- l'imputation, la standardisation et les représentations des catégories sont
  ajustées uniquement sur le train du fold ;
- les agrégats transversaux utilisent seulement les variables disponibles dans
  `X` et sont calculés leave-one-out quand nécessaire ;
- aucun target encoding n'est utilisé ;
- les gains sont comparés ligne par ligne à la baseline, avec une barre d'erreur
  calculée entre dates ;
- les petits gains sont vérifiés sur plusieurs folds et parfois plusieurs seeds ;
- un screening positif ne suffit pas : le modèle doit passer une validation de
  confirmation avant de produire un ensemble ou une soumission.

Cette procédure a notamment permis de détecter plusieurs faux positifs. Par
exemple, le modèle hiérarchique H15 gagnait 0,074 point sur quatre folds, mais
perdait 0,020 point lors de la confirmation sur huit folds.

## 2. Les découvertes les plus importantes

### Il existe un signal, mais il est faible

Les meilleurs modèles locaux atteignent environ 52,5 % d'accuracy et une AUC
proche de 0,535. Le problème n'est donc pas aléatoire, mais le signal exploitable
reste petit par rapport aux variations entre dates.

### `RET_1` porte l'essentiel du signal

Le rendement le plus récent est systématiquement la variable la plus utile. Les
autres returns ajoutent un peu d'information, mais les banques de centaines de
transformations produisent surtout des versions corrélées du même signal.

### Les allocations ont des comportements différents

Les effets fixes d'`ALLOCATION`, ainsi que la pente
`ALLOCATION × RET_1`, améliorent les modèles linéaires. Nous avons aussi trouvé
des styles d'allocation reproductibles et des réactions différentes aux états
de marché. Ces relations sont toutefois instables : elles ne sont utiles
qu'avec une très forte régularisation.

### L'amplitude du score est une vraie mesure de confiance

Pour V1, l'accuracy passe d'environ 50,4 % dans le premier décile de confiance à
57,2 % dans le dernier. Cette relation est positive dans les huit folds.

On sait donc relativement bien identifier les observations faciles et
difficiles. En revanche, nous ne savons pas encore utiliser cette confiance
pour choisir de manière fiable entre deux modèles lorsqu'ils ne sont pas
d'accord.

### Les variables non-return décrivent mieux la difficulté que la direction

Le turnover, la volatilité, la dispersion du marché, l'allocation et la taille
des groupes aident à prévoir la probabilité d'erreur. Elles ajoutent beaucoup
moins de signal pour prédire directement le signe.

`SIGNED_VOLUME_1` contient de l'information, mais environ 73,5 % de ses valeurs
sont manquantes. Une représentation séparant disponibilité et valeur observée
est nécessaire. Le petit gain obtenu n'est pas assez stable pour remplacer la
baseline.

### Le taux de prédictions positives n'explique pas tout

Plusieurs modèles prédisent trop de valeurs positives. Réduire ce taux ou
déplacer le seuil ne suffit cependant pas : les modèles ainsi recentrés perdent
souvent en accuracy, même lorsque leur AUC progresse légèrement.

### Les modèles puissants n'ont pas trouvé le signal manquant

Nous avons testé LightGBM classique et linéaire, des forêts très régularisées,
une Quantile Regression Forest, des modèles low-rank et de nombreuses
interactions non linéaires. Ils n'améliorent pas durablement les modèles
linéaires fortement régularisés.

Le problème principal semble être la stabilité des relations entre dates, et
non un simple manque de profondeur ou de capacité.

### La CV locale et le leaderboard public sont mal alignés

| Modèle | Accuracy OOF locale | Score public connu |
|---|---:|---:|
| V1 Ridge | 0,524709 | 0,511829 |
| V2 logistique sparse | 0,525001 | environ 0,507 pour la soumission apparentée |
| LightGBM à feuilles linéaires | 0,524442 | **0,512018** |

V2 est la meilleure baseline locale, tandis que LightGBM linéaire est le
meilleur modèle public connu. Il ne faut donc pas choisir des hyperparamètres ou
des poids d'ensemble à partir de différences locales de quelques centièmes de
point.

## 3. Les meilleures features trouvées

### Features directionnelles principales

Ce sont les variables les plus utiles pour prédire directement le signe :

1. **`RET_1`** — de loin la feature individuelle la plus importante ;
2. **`RET_2`, `RET_3` et `RET_4`** — complètent la dynamique très récente ;
3. **`RET_7`, `RET_8`, `RET_9` et `RET_18`** — apportent quelques horizons
   supplémentaires sans charger inutilement le modèle ;
4. **`ALLOCATION`** — effet fixe essentiel pour représenter le biais propre à
   chaque stratégie ;
5. **`GROUP`** — effet fixe plus grossier, utile pour mutualiser l'information ;
6. **`ALLOCATION × RET_1`** — meilleure interaction trouvée : chaque allocation
   peut réagir différemment au rendement récent, avec shrinkage vers la pente
   commune.

Ce petit ensemble constitue le cœur de V1 et V2. Il est plus fiable que les
grandes banques de features automatiques.

### États transversaux utiles mais secondaires

Ces features décrivent ce que font simultanément les autres allocations de la
même date. Elles sont construites uniquement avec `X`, généralement en
leave-one-out :

- **`market_ret1_mean`** — moyenne de `RET_1` dans la date ;
- **`market_ret1_dispersion`** — dispersion transversale de `RET_1` ;
- **`market_ret1_breadth`** — proportion d'observations dont `RET_1` est positif ;
- **`market_short_momentum`** — momentum récent moyen du marché ;
- **`market_recent_volatility`** — volatilité récente moyenne ;
- **`group_ret1_relative`** — performance du groupe relativement au marché ;
- **`group_momentum_relative`** — momentum relatif du groupe ;
- **`group_dispersion_relative`** — dispersion du groupe relativement au marché ;
- **résidu de `RET_1` au marché** — `RET_1 - market_ret1_mean` ;
- **résidu de `RET_1` au groupe** — `RET_1 - group_ret1_mean`.

Les plus prometteuses pour une représentation compacte sont
`market_ret1_mean`, `market_ret1_dispersion` et `group_ret1_relative`. Elles
améliorent parfois l'AUC et permettent de décrire la réaction des allocations,
mais leur gain d'accuracy n'est pas suffisamment stable pour les intégrer à la
baseline.

### Features de confiance et de difficulté

Ces variables prédisent davantage **quand** le modèle sera fiable que le signe
lui-même :

- **`abs(score)`** ou amplitude de la prédiction — meilleure mesure de confiance ;
- amplitude de **`RET_1`** ;
- **`MEDIAN_DAILY_TURNOVER`** ;
- volatilité récente de la ligne ;
- dispersion et taille de la date ;
- identité de l'`ALLOCATION` ;
- écart entre les scores de V1 et de LightGBM ;
- maximum des amplitudes des deux scores.

L'amplitude du score est la découverte la plus robuste de cette famille : les
prédictions extrêmes sont nettement plus exactes que les prédictions proches de
zéro. Ces features sont utiles pour le diagnostic, la calibration et
l'abstention, mais notre sélecteur de modèles n'est pas encore assez fiable.

### Features intéressantes mais fragiles

- **`SIGNED_VOLUME_1` observé**, accompagné obligatoirement d'un indicateur de
  disponibilité ;
- rang et amplitude de `SIGNED_VOLUME_1` quand il est présent ;
- profil moyen de `RET_1` par allocation ;
- corrélation historique de l'allocation avec le marché ou son groupe ;
- momentum court et turnover médian du profil d'allocation ;
- transformations centrées sur zéro comme signed-log, `tanh` ou `arctan`.

Elles contiennent parfois du signal, mais les gains changent selon les folds.
Elles ne doivent pas être ajoutées en bloc. `SIGNED_VOLUME_1`, notamment, est
manquant sur environ 73,5 % des lignes et doit être traité comme une variable en
deux parties : disponibilité puis valeur conditionnelle.

## 4. Les meilleures manières trouvées pour modéliser ces données

### Option 1 — Ridge sur la cible continue

La méthode la plus simple et la plus robuste est :

1. utiliser les huit returns principaux ;
2. encoder `GROUP` et `ALLOCATION` en effets fixes ;
3. ajouter une pente `ALLOCATION × RET_1` ;
4. ajuster une Ridge fortement régularisée sur la cible continue ;
5. prendre le signe de la prédiction.

Cette méthode correspond à V1. Elle a une bonne stabilité et obtient l'un des
meilleurs scores publics.

### Option 2 — Régression logistique sparse

V2 utilise la même représentation, mais prédit directement le signe avec une
logistique L2 fortement régularisée (`C=0.003`). C'est notre meilleure baseline
locale avec 0,525001 d'accuracy OOF.

Elle doit rester la référence pour évaluer toute nouvelle feature ou tout
nouveau modèle.

### Option 3 — LightGBM à feuilles linéaires

LightGBM avec `linear_tree=True`, peu profond et fortement régularisé, est notre
meilleure ancre non linéaire. Ses feuilles ajustent des régressions locales au
lieu de prédictions constantes.

Il ne bat pas V2 en CV locale, mais il obtient le meilleur score public connu.
Il doit être conservé comme modèle complémentaire, sans optimiser un blend sur
le leaderboard.

### Extension prudente — réactions conditionnelles des allocations

H11 ajoute des interactions entre les allocations et quelques états de marché
construits uniquement avec `X`. Son gain moyen sur cinq seeds est de seulement
+0,011 point et son intervalle d'incertitude contient zéro.

Cette représentation est intéressante pour comprendre les données et pour une
future diversification, mais elle ne constitue pas une nouvelle baseline.

## 5. Recommandation finale

Pour ces données, la meilleure stratégie est de conserver un petit portefeuille
de modèles simples et très régularisés :

1. **V1 Ridge** pour la cible continue et la robustesse publique ;
2. **V2 logistique sparse** comme meilleure baseline locale ;
3. **LightGBM à feuilles linéaires** comme complément non linéaire et meilleure
   ancre publique.

Les nouvelles idées doivent apporter une information ou une invariance vraiment
nouvelle. Ajouter automatiquement des transformations, de la profondeur ou des
interactions corrélées augmente surtout la variance.

Tout nouveau candidat devrait être évalué avec :

- une CV par dates ;
- un gain apparié et sa barre d'erreur ;
- une confirmation sur huit folds ;
- plusieurs seeds si le gain est inférieur à 0,1 point ;
- aucune sélection de seuil, de seed ou de poids après observation du
  leaderboard.

À ce stade, aucune nouvelle expérience ne justifie de remplacer V1, V2 ou
LightGBM linéaire.
