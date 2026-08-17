# 11 - Score public, biais positif et correlation des erreurs

## Alerte initiale

La soumission V2 produit 60.23 % de classes positives sur le test, alors que la
cible train est positive dans 50.72 % des cas. Le score public rapporte par
l'utilisateur est proche de 50 % contre 52.50 % OOF.

## Calibration de la V2

Le biais positif n'apparait pas uniquement dans le test :

| Mesure | OOF | Test |
|---|---:|---:|
| Probabilite moyenne | 0.50721 | 0.50643 |
| Taux de predictions positives | 0.60603 | 0.60226 |
| Mediane des probabilites | 0.50642 | 0.50600 |

Les distributions de scores OOF et test sont presque identiques. Les
probabilites sont tres concentrees autour de 0.5 : un faible decalage de
calibration transforme donc beaucoup de lignes en classe 1.

Forcer le taux positif OOF vers celui de la cible ne resout pas le probleme :

| Taux positif impose | Seuil OOF | Accuracy OOF |
|---:|---:|---:|
| 50.0 % | 0.50642 | 0.52414 |
| 50.72 % | 0.50597 | 0.52410 |
| 55.0 % | 0.50341 | 0.52446 |
| Seuil naturel, 60.60 % | 0.50000 | **0.52500** |

L'optimum global OOF reste environ 0.5001. Reequilibrer arbitrairement la
soumission vers 50 % serait donc une hypothese de shift, pas une correction
validee.

La V2 a une AUC OOF de 0.5352, stable entre 0.5297 et 0.5441 selon les folds.
Sa balanced accuracy vaut 0.5235. Elle classe donc un peu mieux que le hasard,
mais son signal est faible et sa calibration de classe est asymetrique.

## Correlation des erreurs OOF

| Paire avec V2 | Correlation des erreurs | Desaccord des predictions | Accuracy oracle de la paire |
|---|---:|---:|---:|
| Ridge V1 | 0.7606 | 0.1194 | 0.5846 |
| LightGBM lineaire | **0.6306** | **0.1843** | **0.6169** |
| Specialiste volume | 0.9022 | 0.0488 | 0.5495 |
| Forme quantile SV | 0.8716 | 0.0640 | 0.5570 |

Le specialiste et les transformations quantiles sont presque des variantes de
la V2 : leurs erreurs sont trop correlees pour corriger une chute de 2.5 points
sur le public. LightGBM apporte davantage de diversite, mais son accuracy
individuelle plus faible rend la selection du bon modele difficile. L'oracle
est uniquement un plafond theorique et ne peut pas etre utilise en test.

## Shift observable entre train et test

- aucune nouvelle allocation ou valeur de `GROUP` dans le test ;
- distance de variation totale d'environ 7.4 % pour la distribution des
  allocations et 7.3 % pour celle des groupes ;
- `SIGNED_VOLUME_1` manque sur 73.52 % du train et 75.04 % du test ;
- les principales distributions numeriques ne montrent que des shifts
  moderes ;
- en revanche, le train contient 2 522 dates avec 209 lignes en moyenne, contre
  seulement 120 dates et 266 lignes en moyenne dans le test. La statistique KS
  sur la taille des dates vaut 0.75.

Le shift le plus visible est donc la structure des dates et la composition des
categories, pas la distribution marginale des probabilites V2.

## Interpretation

Le taux positif de 60 % est un signal d'alerte pratique, mais il n'explique pas
a lui seul le score public : le meme comportement existe en OOF et y reste
legerement optimal. La chute publique suggere surtout que le CV par dates
aleatoires surestime la transferabilite de `P(y | X)` vers les 120 dates de
test, ou que les effets fixes de categories ne se transportent pas aussi bien
dans leur nouvelle composition.

## Decision

Avant tout nouveau stacking ou enrichissement de features, il faut auditer la
validation elle-meme :

1. adversarial validation train contre test avec `X` seulement ;
2. construction de folds dont la distribution des tailles de date, groupes et
   allocations ressemble davantage au test ;
3. mesure de V2, Ridge et LightGBM sur ces folds de stress ;
4. soumission alternative a taux positif plus equilibre uniquement si cette
   validation de stress la justifie.

Le script reproductible est `gpt/error_correlation_audit.py` et ses sorties
sont dans `gpt/outputs/error_correlation_audit/`.
