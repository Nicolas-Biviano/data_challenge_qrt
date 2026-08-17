# H4-G — sélection atomique, stable et résiduelle

## Pourquoi remplacer la sélection par blocs

H4-F sélectionnait quatorze groupes. Certains étaient des singletons, mais les
groupes de rang contenaient sept transformations fortement corrélées de la même
variable. Une sélection du bloc entier permet de tester une petite base
fonctionnelle, mais ne répond pas à la question : quelle transformation porte
réellement le signal et laquelle est la plus stable ?

H4-G sépare donc deux notions :

- une **famille de redondance** regroupe des transformations d'une même
  coordonnée ;
- une **feature candidate** est toujours une colonne atomique.

Les familles servent uniquement à empêcher plusieurs copies quasi équivalentes
d'entrer ensemble dans le modèle. Elles ne sont jamais ajoutées comme blocs.

## Candidats préenregistrés

Les 42 colonnes H4 déjà construites sont réparties en huit familles :

1. z-score de `RET_1` par date : 4 transformations ;
2. z-score de `RET_1` par `date × GROUP` : 4 transformations ;
3. rang de `RET_1` par date : 7 transformations ;
4. rang de `RET_1` par `date × GROUP` : 7 transformations ;
5. rang de `TURNOVER` par date : 7 transformations ;
6. rang de `TURNOVER` par `date × GROUP` : 7 transformations ;
7. disponibilité de `SIGNED_VOLUME_1` : 1 indicateur ;
8. part de volumes signés positifs : 5 transformations.

Aucune nouvelle transformation ne sera ajoutée après observation des scores.

## Choix du représentant stable

Dans chaque fold externe, les trois folds internes comparent séparément toutes
les features atomiques au socle V2. Pour chaque feature et chaque régime, on
calcule les trois gains d'accuracy appariés. La stabilité est résumée par :

`score_stable = min(moyenne(gain_global) - SE_global,
                    moyenne(gain_complet) - SE_complet)`

Le représentant d'une famille est la feature qui maximise ce score. Ce choix
est refait dans chaque fold externe : aucune information du fold externe ou des
autres folds externes n'est utilisée.

## Forward résiduelle

La sélection commence avec le socle V2. À chaque étape :

1. chacun des huit représentants encore disponibles est ajouté seul ;
2. le modèle logistique complet est réajusté ;
3. le candidat est donc évalué **conditionnellement aux features déjà
   retenues**, ce qui revient à chercher le signal encore absent du modèle ;
4. les candidats sont classés par leur score stable ;
5. le meilleur est accepté seulement si ses gains globaux et dates complètes
   sont positifs et s'ils sont positifs dans au moins deux folds internes sur
   trois, pour chacun des deux régimes ;
6. toute la famille du représentant accepté est ensuite fermée.

La procédure s'arrête au premier rejet ou après trois ajouts. Elle ne promet
pas que les résidus logistiques sont orthogonalisés exactement, mais le
réajustement conjoint mesure bien la valeur prédictive incrémentale de chaque
feature après les précédentes.

## Validation finale

- huit folds externes composés de dates entières, sans ordre temporel supposé ;
- preprocessing réajusté uniquement sur le train de chaque fold ;
- seuil fixé à 0,5 ;
- comparaison OOF appariée face à V2 ;
- IC à 95 % par bootstrap de dates entières, globalement et sur les dates de
  276 lignes ;
- AUC, balanced accuracy, taux positif et fréquence de sélection rapportés ;
- aucun target encoding.

Le fold externe mesure l'ensemble de la procédure, y compris le choix du
représentant. Le score reste donc honnête malgré le screening supervisé réalisé
dans les folds internes.
