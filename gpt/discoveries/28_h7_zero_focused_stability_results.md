# H7 — résultats de la stability selection zero-focused

## Conclusion

La sélection classique retrouve des transformations extrêmement stables, mais
elles n'ajoutent pas de signal incrémental au modèle qui contient déjà les
returns bruts. H7 obtient 0,524519 d'accuracy contre 0,525001 pour V2 et échoue
son gate.

## Stabilité de la sélection

Fréquence parmi les 64 ajustements Elastic Net internes :

| Candidat | Fréquence top 8 | Fréquence non nulle |
|---|---:|---:|
| `RET_1__zero_tanh050` | 0,9063 | 0,9063 |
| `RET_2__zero_signed_sqrt` | 0,7656 | 0,7656 |
| `RET_18__zero_tanh050` | 0,6250 | 0,6250 |
| `RET_9__zero_signed_sqrt` | 0,4844 | 0,4844 |
| `RET_1__zero_tanh025` | 0,3281 | 0,3438 |

`RET_2 signed-sqrt` est retenue dans les huit folds externes. Une forme `tanh`
de `RET_1` est également retenue dans les huit folds. La sélection est donc
beaucoup plus reproductible que H4-F/H4-G.

## Performance externe

| Modèle | Accuracy | Gain vs V2 | Bootstrap 95 % du gain |
|---|---:|---:|---:|
| V2 | 0,525001 | 0 | — |
| H7 zero stability | 0,524519 | -0,000482 | [-0,001197 ; 0,000209] |

H7 bat V2 dans seulement 4 folds sur 8. Dans la moitié des lignes au plus faible
margin V2, l'accuracy passe de 0,510080 à 0,509188. Les transformations ne
résolvent donc pas la zone ambiguë.

## Interprétation

La stability selection répond correctement à la question « quelle forme reçoit
un coefficient reproductible ? », mais pas nécessairement à « quelle forme
ajoute un signal au modèle de base ? ». `tanh(RET_1)` et `signed_sqrt(RET_2)`
sont fortement corrélées aux returns bruts déjà présents. Elles peuvent être
sélectionnées à chaque sous-échantillon tout en ne faisant que redistribuer la
régularisation entre deux représentations du même signal.

Le résultat confirme l'intuition sur la redondance : la fréquence de sélection
seule ne suffit pas. Une sélection résiduelle conditionnelle serait requise,
mais H4-G l'a déjà testée sans gain stable.

## Décision

- gate H7 échoué ;
- aucune soumission ;
- ne pas élargir la banque de transformations zero-focused ;
- conserver la sélection stable comme diagnostic de forme, pas comme modèle.

