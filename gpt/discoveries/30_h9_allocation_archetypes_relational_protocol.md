# H9 — archétypes d'allocations et profils relationnels de date

## Motivation

H8 établit que les pentes return–target changent réellement selon la date, mais
les moments marginaux du profil H5 ne permettent pas de prévoir leur signe. Une
pente est une covariance. H9 remplace donc le portrait marginal par une
description relationnelle de la cross-section et ajoute des archétypes
comportementaux d'allocations.

Les archétypes reçoivent des descriptions inspirées de la vie réelle uniquement
pour faciliter le raisonnement. Ils ne prétendent pas identifier la nature
réelle des allocations anonymisées.

## Archétypes X-only

Dans chaque fold externe, les allocations sont décrites avec les seules dates
du train externe :

- volatilité, amplitude et moyenne des huit returns V2 ;
- corrélations entre lags, notamment `RET_1`–`RET_2` ;
- turnover et rang de turnover ;
- disponibilité et amplitude des volumes ;
- rang cross-sectionnel de `RET_1` ;
- exposition au facteur moyen `RET_1` de la date.

Après imputation, standardisation et PCA à dix composantes, un KMeans fixé à six
clusters (`n_init=20`, `random_state=0`) produit les archétypes. Aucune target
n'intervient. On mesure la stabilité des partitions entre folds par Adjusted
Rand Index et leur recouvrement avec `GROUP` par Adjusted Mutual Information.

## Profil relationnel d'une date

Le profil H9 concatène :

1. le profil marginal H5 ;
2. les 28 corrélations entre les huit returns V2 ;
3. les valeurs propres de leur matrice de corrélation, son rang effectif et la
   corrélation absolue moyenne ;
4. les corrélations de chaque return avec le turnover et son volume de même lag ;
5. par `GROUP` fourni et par archétype appris : effectif, moyennes et dispersions
   des huit returns, turnover et disponibilité du volume.

Les archétypes sont réappris dans chaque train externe. Une allocation absente
du train externe serait placée dans un cluster supplémentaire `UNKNOWN` ; ce
cas est comptabilisé.

## Modèles pré-enregistrés

Pour chacun des huit targets de pente H8 :

- **Ridge relationnel**, `alpha=100`, prédit la pente rétractée continue ;
- **logistique inversion**, `C=0.01`, `class_weight=balanced`, prédit directement
  si la pente aura le signe opposé à la pente globale du train externe.

Imputation et standardisation sont réapprises dans chaque fold. Aucun PCA n'est
appliqué au profil de date final : Ridge et la régularisation logistique doivent
gérer directement la redondance.

## Validation et gate

H9 réutilise les huit folds de dates H8. Les paramètres de shrinkage et la pente
globale sont recalculés sans les dates de validation.

Une feature passe seulement si :

- la corrélation OOF de pente dépasse 0,10 ;
- l'AUC OOF de la logistique d'inversion dépasse 0,55 ;
- le Ridge améliore la MAE constante dans au moins 6 folds sur 8 ;
- l'AUC logistique validation reste au moins à 70 % de son excès au-dessus de
  0,5 observé au train, afin d'écarter un régime d'overfit manifeste.

Même en cas de gate positif, H9 ne crée pas de soumission. Il autorise seulement
une expérience ultérieure d'interactions à pentes adaptatives.

