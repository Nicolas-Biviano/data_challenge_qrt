# H11 — robustesse à cinq seeds de CV

V2 et H11 ont été gelés puis recalculés sur cinq découpages différents des
dates. Aucun seed n'est sélectionné.

| Seed | V2 | H11 | Gain H11 | Folds gagnés |
|---:|---:|---:|---:|---:|
| 0 | 0,525001 | 0,525318 | +0,000317 | 6/8 |
| 1 | 0,525798 | 0,525933 | +0,000135 | 5/8 |
| 2 | 0,524859 | 0,525271 | +0,000412 | 6/8 |
| 3 | 0,525643 | 0,525715 | +0,000072 | 3/8 |
| 4 | 0,524882 | 0,524491 | -0,000391 | 5/8 |

Le gain moyen est **+0,000109**, soit +0,011 point d'accuracy. Quatre seeds sur
cinq sont positifs et 25 folds sur 40 sont gagnés. Le gate directionnel minimal
passe, mais l'IC 95 % apparié après moyenne des répétitions vaut
**[-0,000681 ; 0,000899]**.

La conclusion correcte est donc : la direction du gain H11 est assez fréquente
pour justifier une nouvelle représentation, mais son amplitude est beaucoup
plus faible que ne le suggérait le seul seed 0. H11 reste un candidat de
diversification et non une amélioration établie.

Changer de seed a ainsi rempli son rôle diagnostique. Choisir le seed 2 aurait
artificiellement annoncé +0,041 point ; choisir le seed 4 aurait conclu à une
perte. Seule la moyenne gelée est interprétable.
