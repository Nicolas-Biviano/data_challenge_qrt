# H8 — inversions de pente par régime de date

## Hypothèse

Une feature peut sembler faible globalement parce que sa relation avec la
target change de signe selon le régime. H8 cherche d'abord à distinguer une
véritable variation de pente du bruit d'échantillonnage, puis à prédire cette
variation avec des caractéristiques de date X-only.

Les dates anonymisées ne sont pas ordonnées : H8 étudie des régimes de date,
pas une dynamique temporelle.

## Pentes étudiées

Pour les huit returns de V2, on calcule dans chaque date la corrélation
cross-sectionnelle entre la feature et la target continue. La corrélation est
transformée par Fisher :

`z_jt = arctanh(corr_jt)` avec variance d'échantillonnage `1 / (n_t - 3)`.

Un modèle à effets aléatoires empirique rétracte ces estimations vers la pente
globale de la feature. Une inversion est définie par une pente rétractée de
signe opposé à la pente globale.

## Test préalable de réalité

Les allocations de chaque date sont séparées déterministiquement en deux
moitiés. La pente est estimée indépendamment dans chaque moitié. Pour chaque
feature on rapporte :

- corrélation de Pearson et Spearman entre les deux estimations ;
- accord de signe ;
- accord de signe parmi les pentes dépassant une erreur standard.

La prédiction des régimes n'est considérée interprétable que pour les features
dont la corrélation split-half de Pearson est positive et dont l'accord de signe
est supérieur à 52 %. Les autres résultats restent descriptifs.

## Prédiction X-only

Chaque date est décrite par le profil X-only déjà utilisé dans H5 : taille,
moments et missingness des returns/volumes/turnover, composition en `GROUP`.

Dans huit folds externes de dates, un Ridge fixé (`alpha=100`) prédit la pente
rétractée de chaque feature. Imputation, standardisation et PCA à 15 composantes
sont réapprises sur les dates du train externe uniquement.

Mesures :

- corrélation et R² de la pente prédite ;
- MAE contre un prédicteur de pente globale constante ;
- AUC et balanced accuracy pour l'inversion de signe ;
- stabilité par fold.

H8 ne passe que si une feature reproductible obtient une corrélation OOF
supérieure à 0,10, une AUC inversion supérieure à 0,55 et une amélioration de
MAE dans au moins 6 folds sur 8. Une interaction de pente adaptative ne sera
construite que dans une expérience ultérieure si ce gate est franchi.

