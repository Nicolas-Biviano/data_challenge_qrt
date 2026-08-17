# H8 — résultats des inversions de pente par date

## Conclusion

Les changements de relation sont réels et très reproductibles entre deux
moitiés disjointes d'allocations d'une même date. En revanche, le profil X-only
actuel, principalement composé de moments marginaux, ne permet pas de prédire
ces régimes. H8 ne passe pas son gate, mais révèle une structure beaucoup plus
forte que les précédents diagnostics de date.

## Réalité des variations de pente

| Feature | Corrélation split-half | Accord de signe | Accord si les deux moitiés dépassent 1 SE | Taux d'inversion rétracté |
|---|---:|---:|---:|---:|
| RET_1 | 0,7926 | 0,7946 | 0,9355 | 0,3779 |
| RET_18 | 0,7572 | 0,7811 | 0,9424 | 0,4956 |
| RET_2 | 0,7770 | 0,7871 | 0,9446 | 0,4802 |
| RET_3 | 0,7781 | 0,7803 | 0,9391 | 0,4897 |
| RET_4 | 0,7826 | 0,7776 | 0,9373 | 0,4742 |
| RET_7 | 0,7722 | 0,7795 | 0,9378 | 0,4766 |
| RET_8 | 0,7846 | 0,8041 | 0,9469 | 0,4603 |
| RET_9 | 0,7727 | 0,7795 | 0,9338 | 0,4758 |

Les huit features passent largement le gate de reproductibilité. Parmi les
dates où chaque moitié fournit une pente supérieure à une erreur standard,
l'accord de signe dépasse 93 %. Les inversions ne sont donc pas une illusion
créée par 65 à 276 observations par date.

La pente globale masque effectivement des relations opposées : sauf pour
`RET_1`, le taux d'inversion est proche de 50 %.

## Prédiction avec le profil X-only H5

| Feature | Corrélation pente prédite/réelle | R² | AUC inversion | Folds avec MAE améliorée |
|---|---:|---:|---:|---:|
| RET_1 | 0,0635 | -0,0026 | 0,5229 | 3/8 |
| RET_18 | 0,0336 | -0,0031 | 0,5277 | 5/8 |
| RET_2 | -0,0181 | -0,0090 | 0,5070 | 2/8 |
| RET_3 | 0,0295 | -0,0056 | **0,5283** | 3/8 |
| RET_4 | 0,0517 | -0,0006 | 0,5214 | 5/8 |
| RET_7 | -0,0470 | -0,0097 | 0,4907 | 1/8 |
| RET_8 | 0,0260 | -0,0047 | 0,5030 | 4/8 |
| RET_9 | 0,0202 | -0,0064 | 0,5162 | 4/8 |

Tous les R² sont négatifs et aucune feature n'atteint les seuils pré-enregistrés
de corrélation 0,10, AUC 0,55 et amélioration MAE dans 6 folds.

## Interprétation

Le profil H5 décrit surtout les distributions marginales : moyenne, dispersion,
missingness et composition de groupe. Or une pente cross-sectionnelle est une
relation entre deux variables. Le régime manquant peut donc être porté par la
géométrie multivariée de `X` : corrélations entre lags, structure de facteurs,
dispersion sectorielle ou alignement entre returns et volumes.

H8 distingue ainsi clairement deux questions :

1. la pente change-t-elle réellement ? **Oui, fortement** ;
2. savons-nous prévoir ce changement avec le profil actuel ? **Non**.

## Décision

- aucun modèle adaptatif ni soumission à partir de H8 ;
- ne pas interpréter l'échec du Ridge comme absence de régimes ;
- prochaine hypothèse recommandée : construire un profil de date X-only fondé
  sur les covariances/corrélations cross-sectionnelles et les composantes
  factorielles, puis reprendre exactement la même validation H8.

