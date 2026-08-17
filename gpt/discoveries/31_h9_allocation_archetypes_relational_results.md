# H9 — résultats des archétypes et du profil relationnel

## Conclusion

Les profils X-only produisent des archétypes d'allocations stables et largement
différents des quatre `GROUP` fournis. Cette représentation est utile pour
raisonner sur les allocations. Le modèle de régime construit sur 290 variables
de date sur-apprend toutefois fortement et ne prédit pas les inversions H8 hors
date. Aucun gate n'est franchi.

## Archétypes comportementaux

La stabilité moyenne des partitions entre folds est `ARI = 0,878`. Le
recouvrement avec `GROUP` est faible (`AMI = 0,126`). Les clusters ne sont donc
pas une simple reconstruction de `GROUP`.

Description prudente des centroïdes calculés sur tout le train :

| Archétype | Effectif | Description X-only dominante |
|---|---:|---|
| 0 | 87 | faible volatilité, lags persistants, `RET_1` aligné avec le volume |
| 1 | 66 | forte volatilité et forte exposition au facteur moyen de date |
| 2 | 103 | lags plutôt opposés/reversing, volume moins disponible |
| 3 | 9 | turnover et disponibilité volume très élevés, comportement extrême |
| 4 | 2 | outliers de volatilité extrême et volume très peu disponible |
| 5 | 11 | turnover extrême, returns passés élevés, faible alignement au facteur |

Ces noms facilitent la réflexion mais ne décrivent pas la véritable identité
économique des allocations anonymisées. Les clusters 3 à 5, surtout le cluster
4, montrent également une faiblesse du KMeans : quelques outliers forment leurs
propres archétypes.

## Prédiction des régimes

| Feature | Corrélation Ridge OOF | AUC logit OOF | AUC logit train moyenne | Folds MAE gagnants |
|---|---:|---:|---:|---:|
| RET_1 | 0,0305 | 0,5199 | 0,6948 | 0/8 |
| RET_18 | 0,0407 | 0,5179 | 0,6920 | 0/8 |
| RET_2 | 0,0448 | 0,5062 | 0,6911 | 0/8 |
| RET_3 | 0,0449 | 0,4979 | 0,6908 | 0/8 |
| RET_4 | 0,0443 | **0,5435** | 0,7055 | 0/8 |
| RET_7 | 0,0407 | 0,5295 | 0,6960 | 1/8 |
| RET_8 | **0,0672** | 0,5122 | 0,6882 | 0/8 |
| RET_9 | -0,0289 | 0,4988 | 0,6881 | 0/8 |

Tous les R² Ridge sont négatifs. Le modèle constant donne une MAE inférieure
dans 63 folds feature × fold sur 64.

Le contraste train-validation est net :

- corrélation Ridge train moyenne : environ 0,36–0,39 ;
- corrélation Ridge validation : -0,03 à 0,07 ;
- AUC logit train : 0,688–0,706 ;
- AUC logit validation : 0,498–0,544.

H9 est donc limité par la variance et une régularisation insuffisante pour 290
variables et seulement 2 522 dates. Il ne faut pas conclure que la structure
relationnelle est inutile.

## Comparaison H8–H9

Le profil relationnel n'améliore pas systématiquement le profil marginal H8.
`RET_4` progresse en AUC de 0,5214 à 0,5435, mais son excès d'AUC validation ne
représente que 21 % de l'excès observé au train. `RET_7` progresse également,
alors que plusieurs autres features se dégradent.

Cette hétérogénéité et le fort écart train-validation interdisent de retenir les
deux améliorations apparentes.

## Décision

- conserver les archétypes comme outil descriptif ;
- ne construire aucune pente adaptative et aucune soumission ;
- ne pas ajouter davantage de caractéristiques relationnelles ;
- si H9 est poursuivi, effectuer une nouvelle expérience H9-B séparée avec
  sélection imbriquée de régularisations beaucoup plus fortes et clustering
  robuste aux outliers.

