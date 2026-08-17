# H15 — Résultats de la hiérarchie basse dimension

## Screening 4-fold

Le screening était positif. `hierarchical_strong`, gelé avec trois états,
`group_scale=0,30` et `allocation_scale=0,10`, obtenait :

- baseline : 0,524648 ;
- challenger : 0,525386 ;
- gain : **+0,000738** ;
- AUC : 0,535505 contre 0,535296 ;
- gap train–validation : 0,00534.

Le gate autorisait donc la confirmation 8-fold.

## Confirmation 8-fold

Le gain ne se confirme pas :

| Mesure | V2 | H15 hiérarchique |
|---|---:|---:|
| Accuracy OOF | **0,525001** | 0,524806 |
| AUC moyenne des folds | 0,535416 | **0,535771** |
| Taux positif | 60,60 % | 60,26 % |
| Accuracy train moyenne | 0,529491 | 0,530344 |

Le gain d'accuracy vaut **−0,000195**, avec une erreur standard appariée par
date de 0,000485 et un IC95 `[-0,001147 ; +0,000756]`. H15 gagne seulement trois
folds sur huit.

| Fold | Gain H15 − V2 |
|---:|---:|
| 1 | +0,000530 |
| 2 | −0,001522 |
| 3 | −0,000077 |
| 4 | −0,002029 |
| 5 | −0,000668 |
| 6 | +0,001268 |
| 7 | −0,001133 |
| 8 | +0,002066 |

## Interprétation

La réduction à trois états évite le fort recul de la hiérarchie H11 à huit
états. Elle améliore même légèrement le classement AUC. Mais la réaction
hiérarchique n'améliore pas le signe au seuil fixé et son avantage 4-fold était
un faux positif de validation.

Le gap train–validation reste modéré, proche de 0,55 point. Le problème n'est
pas un surapprentissage violent ; les petites réponses conditionnelles changent
suffisamment entre les sous-ensembles de dates pour annuler le gain.

## Décision

- ne pas ajuster les échelles `0,30/0,10` après lecture des huit folds ;
- ne pas calibrer un seuil sur les mêmes OOF ;
- ne pas lancer le multi-seed, réservé par protocole aux confirmations positives ;
- ne pas empiler H15 avec V2 ;
- conserver H11/H15 comme preuve d'un signal de ranking conditionnel faible,
  pas comme candidat d'accuracy.
