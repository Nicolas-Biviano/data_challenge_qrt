# 06 - Plan experimental V3

## Objectif

Depasser la V2 logistique (accuracy OOF 0.525001, score penalise TS 0.523285)
avec des modeles capables de combiner partitions non lineaires et relations
localement lineaires.

## Contraintes inchangees

- aucun ordre suppose entre les labels de date;
- folds construits sur des `TS` complets;
- aucun target encoding;
- aucune reconstruction des fenetres temporelles;
- preprocessing ajuste dans le train du fold;
- toutes les decisions de seuil ou de blend sont prises hors fold.

## Blocs de features

1. `returns`: rendements bruts selectionnes et resumes robustes;
2. `non_return`: volumes signes, turnover, missingness et statistiques
   transversales;
3. `combined`: union des deux blocs et categories construites uniquement avec
   `X`.

Les clusters d'allocation seront appris a partir de profils de volumes,
turnover et missingness. Pour chaque fold, le profil et le clustering seront
ajustes uniquement sur les dates d'entrainement.

## Ordre des experiences

1. LightGBM regression a feuilles constantes;
2. LightGBM regression avec `linear_tree=True`;
3. Quantile Regression Forest: moyenne, mediane et intervalles;
4. Cubist/model tree seulement si LightGBM lineaire est prometteur;
5. Local Linear Forest seulement si les model trees franchissent le gate.

## Gates

### Screening

Deux folds fixes. Un candidat doit battre la V2 sur la moyenne des memes lignes
et ne pas perdre lourdement sur l'un des deux folds.

### Validation complete

Pour etre retenu, un modele doit viser:

- accuracy au moins egale a 0.5260;
- score penalise TS superieur a 0.523285;
- gain sur au moins cinq folds sur huit;
- comportement stable sur un second `random_state`;
- distributions train/test et taux de classes documentes.

## Regle anti-rabbit-hole

Une branche qui echoue au screening est documentee puis arretee. Les grilles
sont petites et definies avant execution. Ensemble et stacking n'interviennent
qu'apres la validation de modeles autonomes.

