# H5 — estimateur par panels test-sized et confiance d'amplitude

## Objectif

La CV actuelle répond à la performance moyenne sur une date train aléatoire.
Le test contient seulement 120 dates et son profil X est très différent. H5
construit donc une distribution de validations de 120 dates, appariées aux 120
dates test avec X uniquement.

H5 ne sert pas à créer de nouvelles features ni à ajuster un modèle. Les trois
prédictions OOF sont figées avant l'expérience : V1 Ridge, V2 logistique et
LightGBM à feuilles linéaires.

## Profil et appariement des dates

Chaque date est décrite par :

- son nombre de lignes et d'allocations ;
- les moyennes et écarts-types des returns sélectionnés, volumes et turnover ;
- les taux de valeurs manquantes des volumes ;
- la composition en `GROUP`.

Les valeurs manquantes sont imputées, les colonnes standardisées puis résumées
par 15 composantes principales, apprises sur les profils train et test réunis.
Cette étape est transductive mais strictement X-only.

Pour chaque date test, on calcule ses 30 dates train voisines. Un panel est
obtenu en parcourant les 120 dates test dans un ordre aléatoire et en tirant une
date train encore inutilisée parmi ses voisines, avec une probabilité
décroissante en fonction de la distance. On génère 1 000 panels appariés et
1 000 panels uniformes de contrôle.

## Mesures de l'estimateur

Pour chaque panel et modèle :

- accuracy pondérée par ligne, correspondant à la métrique publique ;
- moyenne des accuracies par date ;
- taux positif prédit ;
- différence appariée LightGBM − V1 et V2 − V1.

Les quantiles de la distribution des panels mesurent la sensibilité au choix
des dates. Ils ne constituent pas un intervalle de confiance classique : les
panels se recouvrent et l'hypothèse de covariate shift peut être fausse.

Le diagnostic principal est la probabilité empirique qu'un modèle batte un
autre et la fréquence du classement public connu
`LightGBM > V1 > logistique`.

## Confiance issue de l'amplitude

Dans chaque fold OOF et pour chaque modèle de régression, les lignes sont
classées par percentile de `|score prédit|`, puis regroupées en dix déciles.
On mesure :

`P(signe correct | décile de |score|)`.

La normalisation à l'intérieur du fold empêche une différence d'échelle entre
folds de créer artificiellement la courbe. La stabilité est rapportée fold par
fold. Un sélecteur diagnostique non entraîné choisit, uniquement lorsque V1 et
LightGBM sont en désaccord, la prédiction dont le percentile d'amplitude est le
plus élevé. Il ne sera pas soumis sans validation supplémentaire.

## Règles de décision

1. Si les panels appariés ne reproduisent pas mieux le classement public que
   les panels uniformes, ils ne seront pas utilisés pour tuner.
2. Si l'accuracy augmente avec `|score|` dans au moins 7 folds sur 8, l'amplitude
   est acceptée comme mesure de confiance OOF.
3. Aucun nouveau modèle ou seuil n'est choisi dans H5.
