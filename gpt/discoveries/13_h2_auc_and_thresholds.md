# H2 — AUC et seuils par date

## Hypothèse

Le taux de prédictions positives proche de 60 % pourrait signaler un défaut de
calibration plutôt qu'une absence de classement. On mesure l'AUC, la courbe de
seuil globale, un oracle par date et des seuils appris à partir de profils de
marché X-only.

## Résultats

- AUC OOF globale : 0,535216 ;
- accuracy au seuil 0,5 : 0,525001 ;
- oracle par date : 0,609109 ;
- offset constant cross-fitté : 0,523144 ;
- offset Ridge marché : 0,521865 ;
- offset Random Forest marché : 0,522618.

L'oracle voit les labels de la date évaluée et sélectionne le meilleur seuil
parmi de nombreux candidats. Son 60,91 % est donc une borne descriptive très
optimiste, pas un score déployable.

## Décision

H2 n'est pas soutenue avec les features de marché actuelles. Les scores ont un
faible pouvoir de classement, mais tous les seuils appris hors échantillon font
moins bien que 0,5. Corriger le taux positif vers 50 % n'est pas une solution.

Notebook : `gpt/notebooks/H2_auc_and_thresholds.ipynb`.
