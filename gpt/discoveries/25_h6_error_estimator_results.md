# H6 — résultats de l'estimateur d'erreur conditionnelle

## Conclusion

H6 améliore réellement l'estimation de la difficulté absolue d'une ligne. Le
modèle compact bat le taux d'erreur constant en Brier et en log-loss dans les
huit folds pour V1 et LightGBM. Il identifie un sous-ensemble de 10 % des lignes
où l'accuracy atteint environ 58,3 %.

Cette amélioration ne se transforme pas en bon sélecteur V1/LightGBM. Sur leurs
désaccords, le classifieur direct a une AUC validation de 0,505 et aucun des
quatre sélecteurs ne passe le gate pré-enregistré.

## Probabilité d'erreur OOF

| Modèle de base | Estimateur | Brier | Log-loss | ROC-AUC erreur |
|---|---|---:|---:|---:|
| V1 | Constant | 0,249394 | 0,691934 | 0,496007 |
| V1 | Amplitude logit | 0,248987 | 0,691118 | 0,521430 |
| V1 | Compact logit | **0,248853** | **0,690839** | **0,523501** |
| LightGBM | Constant | 0,249406 | 0,691960 | 0,496118 |
| LightGBM | Amplitude logit | 0,249022 | 0,691190 | 0,521197 |
| LightGBM | Compact logit | **0,248864** | **0,690860** | **0,523562** |

Les deux modèles entraînés battent le constant en Brier et en log-loss dans
8/8 folds. Le gain est faible en valeur absolue, mais stable.

L'amplitude logit est mieux calibrée par décile : ECE de 0,00198 pour V1 et
0,00265 pour LightGBM. Le compact discrimine mieux mais son ECE monte à 0,00711
et 0,00676. Il surestime notamment l'erreur dans ses déciles supérieurs. Il ne
doit pas être interprété comme une probabilité parfaitement calibrée sans une
nouvelle couche de calibration cross-fittée.

## Diagnostic biais–variance

| Cible | Estimateur | AUC train moyenne | AUC validation moyenne |
|---|---|---:|---:|
| erreur V1 | Amplitude | 0,521668 | 0,521715 |
| erreur V1 | Compact | 0,531942 | 0,523604 |
| erreur LightGBM | Amplitude | 0,521357 | 0,521392 |
| erreur LightGBM | Compact | 0,531386 | 0,523668 |
| choisir LightGBM sur désaccord | Compact direct | 0,536866 | 0,505277 |

Le compact présente un petit écart train-validation pour la difficulté absolue,
mais conserve du signal. Sur le choix relatif, presque tout le signal train
disparaît : c'est un cas net de variance/relations non transportables, pas une
preuve que le modèle est simplement trop régularisé.

## Abstention diagnostique

Accuracy parmi les lignes ayant la plus faible probabilité d'erreur :

| Base | Estimateur | Top 10 % | Top 20 % | Toutes les lignes |
|---|---|---:|---:|---:|
| V1 | Amplitude | 0,570949 | 0,557852 | 0,524709 |
| V1 | Compact | **0,583623** | **0,562131** | 0,524709 |
| LightGBM | Amplitude | 0,567382 | 0,555661 | 0,524442 |
| LightGBM | Compact | **0,583205** | **0,562965** | 0,524442 |

Ce résultat valide l'estimateur comme outil de diagnostic et de pondération de
confiance. La métrique du challenge exige toutefois une prédiction sur toutes
les lignes : l'abstention ne constitue pas directement une soumission.

## Sélection V1/LightGBM

| Sélecteur | Accuracy OOF | Accuracy sur désaccords | Folds gagnants vs V1 | P(gain > 0), panels appariés |
|---|---:|---:|---:|---:|
| V1 | 0,524709 | 0,500836 | — | — |
| Rang d'amplitude | 0,525242 | 0,504168 | 5/8 | 0,652 |
| Erreur amplitude | **0,525280** | **0,504405** | 6/8 | 0,652 |
| Erreur compacte | 0,524916 | 0,502128 | 6/8 | 0,694 |
| Compact direct | 0,525053 | 0,502982 | 5/8 | 0,725 |

Le gate exigeait au moins 7 folds gagnants et une probabilité de gain d'au
moins 90 % sur les panels appariés. Aucun sélecteur ne passe. La meilleure
accuracy OOF n'est que `+0,000571` au-dessus de V1, avec une décision correcte à
50,44 % sur les désaccords.

## Ce que les features ajoutent

Pour la difficulté absolue, les coefficients numériques stables font surtout
intervenir :

- l'amplitude moyenne de `RET_1` dans la date ;
- le turnover ;
- la volatilité récente ;
- l'amplitude maximale des deux scores ;
- la taille et la dispersion de la date ;
- la différence entre les scores.

Les effets `ALLOCATION` sont souvent les plus grands coefficients. Ils
améliorent la difficulté absolue, mais dominent aussi le classifieur de
désaccord qui ne généralise presque pas. Ils ne sont donc pas une justification
pour construire un nouveau sélecteur plus complexe.

## Décision

1. H6 est retenu comme estimateur de difficulté et outil d'analyse.
2. Aucun sélecteur H6 n'est retenu pour soumission.
3. La prochaine extension admissible est une calibration sigmoid strictement
   cross-fittée du compact, évaluée uniquement sur Brier/log-loss/ECE.
4. Une extension pour le choix relatif ne sera tentée que si elle cible le
   résidu `error_lgbm - error_v1` et retire d'abord les effets communs de
   difficulté ; elle devra rester séparée de la calibration absolue.

