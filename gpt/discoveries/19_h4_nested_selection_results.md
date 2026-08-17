# H4-F/H4-G — résultats des sélections imbriquées

## H4-F : forward par groupes

H4-F a comparé quatorze groupes dans trois folds internes pour chacun des huit
folds externes. Un groupe pouvait contenir une seule feature ou plusieurs
transformations corrélées. Au plus trois groupes étaient ajoutés.

| Mesure | V2 | H4-F | Différence H4-F − V2 |
|---|---:|---:|---:|
| Accuracy globale | 0,525001 | 0,525233 | +0,000231 |
| Accuracy dates complètes | 0,518438 | 0,518759 | +0,000322 |
| AUC | 0,535216 | 0,535990 | +0,000774 |
| Balanced accuracy | 0,523483 | 0,523893 | +0,000410 |
| Taux positif | 0,606030 | 0,593622 | −0,012408 |

L'IC 95 % apparié du gain global est `[-0,000670 ; 0,001127]`. Sur les
dates complètes il vaut `[-0,000860 ; 0,001446]`. Le petit gain moyen n'est donc
pas distinguable du bruit.

Les groupes les plus fréquents étaient l'indicateur de disponibilité de SV1
(6 folds sur 8) et le bloc de rang de `TURNOVER` par date (4 sur 8). Le défaut
de H4-F est que ce dernier ajoutait sept formes corrélées en même temps.

## H4-G : représentant atomique puis signal résiduel

H4-G a comparé 42 colonnes atomiques réparties dans huit familles de
redondance. Un seul représentant par famille était autorisé. Après chaque
ajout, la logistique était réajustée et le candidat suivant était évalué
conditionnellement aux features déjà retenues.

La règle d'acceptation exigeait un gain positif global, un gain positif sur les
dates complètes et au moins deux folds internes positifs sur trois pour les deux
régimes. Le protocole complet est dans
`gpt/discoveries/18_h4_atomic_residual_protocol.md`.

| Mesure | V2 | H4-G | Différence H4-G − V2 |
|---|---:|---:|---:|
| Accuracy globale | 0,525001 | 0,524686 | −0,000315 |
| Accuracy dates complètes | 0,518438 | 0,518056 | −0,000381 |
| AUC | 0,535216 | 0,535322 | +0,000105 |
| Balanced accuracy | 0,523483 | 0,523294 | −0,000189 |
| Taux positif | 0,606030 | 0,597211 | −0,008819 |

L'IC 95 % apparié du gain global est `[-0,001108 ; 0,000423]`. Sur les
dates complètes il vaut `[-0,001335 ; 0,000549]`. H4-G ne bat donc pas V2.
Le calcul complet a pris 709 secondes, soit 11 minutes 49 secondes.

## Ce que la sélection atomique apprend

- la famille du rang de `TURNOVER` par date est retenue dans 4 folds sur 8 ;
- aucune de ses formes atomiques n'est retenue plus d'une fois : `tanh05`,
  `tanh1`, signed-log et hinge 20 gagnent chacune un fold ;
- la famille du rang de `RET_1` par date entre dans 3 folds, mais sa forme reste
  elle aussi instable ;
- `sv1_available`, seule vraie feature atomique parfaitement identifiable, est
  retenue dans 3 folds ;
- un fold externe n'ajoute aucune feature.

La famille `TURNOVER` semble donc contenir un signal faible, mais les données ne
permettent pas d'identifier une transformation universellement meilleure. Le
choix supervisé d'un représentant dans chaque fold interne consomme plus de
variance que le gain qu'il récupère.

## Décision

La procédure résiduelle est méthodologiquement plus propre que l'ajout de blocs
corrélés, mais elle ne transforme pas H4 en amélioration validée. V2 reste le
modèle retenu. Pour une bibliothèque de dizaines de milliers de features, la
suite raisonnable serait un filtrage de redondance X-only, puis une stability
selection sur les représentants ; elle ne doit pas être déclenchée tant que la
bibliothèque de features n'a pas été définie indépendamment de ces résultats.
