# H5 — résultats des panels test-sized et de la confiance d'amplitude

## Résumé

L'appariement X-only produit des panels de 120 dates train beaucoup plus proches
du test que des panels uniformes. Il retrouve aussi plus souvent le classement
public connu. Il ne réduit toutefois pas assez l'incertitude pour distinguer
fiablement des modèles séparés par quelques centièmes de point.

À l'inverse, `|score prédit|` est un indicateur de confiance net et stable :
l'accuracy croît avec son décile dans les huit folds, pour V1 et LightGBM.

## Qualité de l'appariement

Moyennes sur 1 000 panels de chaque type :

| Mesure, plus bas = mieux | Uniforme | Apparié |
|---|---:|---:|
| Distance PCA du profil moyen au test | 2,1743 | 1,0100 |
| Écart standardisé absolu moyen | 0,2160 | 0,1180 |
| Écart standardisé absolu maximal moyen | 1,5490 | 1,0410 |

Le mécanisme d'appariement remplit donc son objectif descriptif. Cela ne prouve
pas que la relation `P(y | X)` soit identique entre les dates train appariées et
le test.

## Scores sur les panels de 120 dates

| Régime | Modèle | Accuracy moyenne | Quantiles empiriques 2,5–97,5 % |
|---|---|---:|---:|
| Apparié | LightGBM linéaire | 0,519246 | [0,509120 ; 0,529680] |
| Apparié | V1 Ridge | 0,518624 | [0,508057 ; 0,529425] |
| Apparié | V2 logistique | 0,518585 | [0,508289 ; 0,529205] |
| Uniforme | LightGBM linéaire | 0,524600 | [0,513246 ; 0,537370] |
| Uniforme | V1 Ridge | 0,525051 | [0,511662 ; 0,539268] |
| Uniforme | V2 logistique | 0,525315 | [0,512062 ; 0,540033] |

Sur les panels appariés, LightGBM − V1 vaut en moyenne `+0,000622`, mais
l'intervalle empirique 95 % est `[-0,005398 ; +0,006316]`. LightGBM gagne dans
57,6 % des panels seulement. V2 − V1 est pratiquement nul (`-0,000039`) et V2
gagne dans 48,3 % des panels.

Le classement public exact `LightGBM > V1 > V2` apparaît dans 25,3 % des panels
appariés, contre 14,9 % des panels uniformes. La corrélation de rang moyenne avec
le public passe de -0,131 à +0,129. C'est une amélioration, pas une validation
suffisante pour tuner.

## Confiance d'amplitude

Accuracy du premier et du dernier décile de `|score|`, calculés à l'intérieur
de chaque fold :

| Modèle | Décile 1 | Décile 10 | Folds à pente positive |
|---|---:|---:|---:|
| V1 Ridge | 0,503757 | 0,572358 | 8/8 |
| LightGBM linéaire | 0,499127 | 0,569209 | 8/8 |

La cible absolue moyenne croît également avec la confiance : de 0,001860 à
0,003287 pour V1 et de 0,001806 à 0,003474 pour LightGBM. Les scores élevés
identifient donc partiellement des rendements futurs de plus grande amplitude,
dont le signe est plus facile à prédire.

## Test du sélecteur de modèle

Sur les 16,0 % de lignes où V1 et LightGBM divergent, sélectionner le modèle au
percentile d'amplitude le plus élevé donne :

- accuracy globale : 0,525242 ;
- V1 seule : 0,524709 ;
- LightGBM seule : 0,524442 ;
- accuracy de la décision sur les seuls désaccords : 0,504168.

Le gain OOF sur V1 n'est que de 0,000533 et la décision sur les désaccords est
quasiment aléatoire. Le sélecteur n'est pas un candidat à soumettre.

## Décision

1. Conserver les panels appariés comme distribution de stress secondaire.
2. Ne pas les utiliser comme score unique de tuning.
3. Conserver `|score|` comme feature d'un futur estimateur d'erreur OOF.
4. Évaluer cet estimateur par log-loss, Brier et calibration, et non seulement
   par l'accuracy finale d'un ensemble.
5. Ne produire aucune nouvelle soumission à partir de H5.

