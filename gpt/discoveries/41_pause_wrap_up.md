# Wrap-up de pause — recherche GPT sur le challenge QRT

## Résumé en une phrase

Nous avons extrait un signal directionnel faible mais réel autour de `RET_1`,
des styles d'allocation et de l'état transversal du marché ; en revanche, aucune
extension n'a démontré un gain suffisamment stable pour remplacer les trois
ancres actuelles — V1 Ridge, V2 logistique et LightGBM linéaire — et la CV locale
classe mal les modèles sur le leaderboard public.

## État des trois ancres

| Modèle | Accuracy OOF | Score public connu | Lecture actuelle |
|---|---:|---:|---|
| V1 Ridge à effets fixes | 0,524709 | 0,511829 | meilleure ancre linéaire simple et robuste |
| V2 logistique sparse | 0,525001 | 0,507060 pour la soumission publique apparentée | meilleure CV locale, transfert public faible |
| LightGBM à feuilles linéaires | 0,524442 | **0,512018** | meilleur public connu, malgré une CV inférieure |

La correspondance exacte entre V2 locale (`C=0.003`) et l'ancienne ligne
publique (`C=0.001`) est imparfaite. Il ne faut donc pas surinterpréter leur
écart. Le blend V1/LightGBM 50/50 gagnait 0,049 point sur V1 en OOF, mais a fait
0,510700 public : même la complémentarité OOF ne s'est pas transférée.

## Ce que nous avons essayé

### 1. Baselines, effets fixes et régularisation linéaire

- Ridge sur la cible continue, puis seuil à zéro ;
- logistique Elastic Net/sparse sur le signe ;
- effets fixes `ALLOCATION` et `GROUP` ;
- pente `ALLOCATION × RET_1` avec shrinkage ;
- calibration du seuil et différentes forces de pénalisation.

Conclusion : `RET_1` domine et les effets d'allocation sont utiles, mais V2 ne
gagne que 0,029 point sur V1 en OOF. Le leaderboard préfère pourtant V1.

### 2. Features brutes, volume et transformations automatiques

- returns retardés et familles non-return ;
- `SIGNED_VOLUME_1` en représentation deux-parties : disponibilité, valeur,
  rangs et interactions ;
- quantiles conditionnels et formes détectées sur les résidus ;
- `tanh`, `arctan`, `arcsin`, signed-log, ratios et transformations centrées sur
  zéro ;
- forward selection groupée, puis atomique/résiduelle et CV imbriquée ;
- stability selection Elastic Net sur 40 formes zero-focused.

Conclusion : `SIGNED_VOLUME_1` est absent sur environ 73,5 % des lignes. Son
spécialiste atteint 0,525185, mais ne gagne que quatre folds sur huit. La
meilleure forme issue des quantiles atteint 0,525024. La forward groupée monte à
0,525233, mais son IC de gain contient zéro ; la sélection atomique retombe à
0,524686. H7 atteint 0,524519. Les transformations créent surtout des versions
corrélées d'un signal déjà présent et augmentent la variance de sélection.

### 3. Validation, seuils et dates difficiles

- entraînement limité aux dates les plus proches du test (H1) ;
- AUC, seuil global, seuil oracle par date et tentative d'apprentissage du seuil
  à partir d'états de marché (H2) ;
- classification des dates faciles/difficiles et analyse fine des lignes où les
  modèles se trompent ensemble (H3) ;
- panels de 120 dates appariés au test uniquement à partir de `X` ;
- répétition de la CV sur cinq seeds.

Conclusion : les dates « test-like » seules tombent à 0,518318. Les seuils
appris restent sous le seuil fixe 0,5. Les dates difficiles ne sont pas
prédictibles à partir de leurs profils marginaux (`AUC=0,4688`) et les lignes
unanimement fausses ne sont que faiblement séparables (`AUC=0,5239`). Les panels
appariés reproduisent mieux le classement public, mais l'intervalle des écarts
entre modèles reste trop large pour tuner.

### 4. Confiance et estimation de l'erreur

- déciles de `|score|` ;
- estimateur cross-fitté de probabilité d'erreur ;
- abstention diagnostique ;
- sélecteurs V1/LightGBM sur leurs désaccords.

Conclusion solide : l'amplitude est une vraie mesure de confiance. Pour V1,
l'accuracy passe de 50,38 % dans le premier décile à 57,24 % dans le dernier,
avec une pente positive dans 8/8 folds. L'estimateur compact d'erreur atteint
une AUC de 0,5235 et isole 10 % de lignes à environ 58,3 % d'accuracy. Mais le
choix relatif entre V1 et LightGBM reste aléatoire (`AUC=0,5053`) : aucun
ensemble conditionnel n'a passé le gate.

### 5. Régimes, dé-factorisation et styles d'allocation

- retrait des composantes marché et `GROUP` des returns ;
- archétypes d'allocation X-only ;
- états leave-one-out de marché et de groupe ;
- fonctions de réponse `ALLOCATION × état` fortement shrinkées ;
- prédiction des inversions de relation entre feature et cible.

Conclusion : les archétypes d'allocation sont reproductibles (`ARI=0,878`) et
les inversions de pente sont réelles (accord de signe split-half proche de
78–80 %). Mais prédire ces inversions à partir de `X` reste faible
(`AUC max=0,5283`) et les modèles relationnels flexibles sur-apprennent. Les
returns dé-factorisés perdent 8/8 folds. H11 est la seule direction faiblement
positive : sur cinq seeds, son gain moyen n'est que **+0,000109** avec un IC95
`[-0,000681 ; +0,000899]`.

### 6. Modèles plus puissants

- HistGradientBoosting classification/régression ;
- LightGBM classique et `linear_tree=True` ;
- Quantile Regression Forest : moyenne et médiane conditionnelles ;
- factorization machine/correction bilinéaire low-rank ;
- Random Forest tunée par CV imbriquée avec forte randomisation, élagage,
  grandes feuilles et 1 000 arbres.

Conclusion : la capacité brute n'a pas récupéré le signal manquant.

| Modèle puissant | Résultat principal | Diagnostic |
|---|---:|---|
| LightGBM linéaire complet | 0,524442 | meilleur public, sous V2 en OOF |
| LightGBM conditionnel H12 | 0,524072 | profondeur réelle 3, gap > 1 point |
| Low-rank rang 4 | 0,519385 sur 2 folds | mémorise rapidement les allocations |
| QRF moyenne / médiane | 0,513228 / 0,510228 sur 1 fold | screening nettement négatif |
| RF H13 très régularisée | 0,523711 | −0,129 point vs V2, 4/8 folds |

La RF H13 a un gap train–validation moyen de seulement 0,415 point et sa courbe
est stabilisée vers 600 arbres. Son échec n'est donc pas un simple overfit que
1 000 arbres supplémentaires corrigeraient : elle perd aussi en AUC et en
Brier. Les arbres constants ont trop de biais ; les modèles plus flexibles
apprennent des relations qui changent entre dates.

## Conclusions solides

1. **Il y a du signal.** V2 a une AUC OOF proche de 0,535 et l'amplitude du
   score ordonne nettement la difficulté.
2. **Le signal principal reste court et linéaire.** `RET_1`, son amplitude, les
   effets d'allocation et quelques états transversaux portent l'essentiel.
3. **Les non-return décrivent mieux la difficulté que la direction.** Turnover,
   volatilité, taille de date et allocation aident à savoir quand le modèle se
   trompera, beaucoup moins à choisir le bon signe ou le bon modèle.
4. **Les styles d'allocation existent**, mais leur réaction future aux états de
   marché n'est exploitable qu'avec un shrinkage très fort ; le gain H11 est
   directionnel, pas établi.
5. **La variance entre dates domine les petits gains.** Un gain de quelques
   centièmes de point sans erreur standard et sans répétition des seeds n'est
   pas une preuve.
6. **La CV locale n'est pas alignée avec le public.** Sur V1, V2 et LightGBM,
   son classement est même inversé. Le leaderboard ne doit pas servir à tuner
   des poids ou des hyperparamètres après coup.
7. **Le taux de positifs est un symptôme, pas l'explication unique.** Le réduire
   n'améliore pas automatiquement l'AUC, le Brier ou l'accuracy.

## Ce que nous arrêtons pendant la pause

- les grandes banques de transformations non linéaires ;
- les recherches de seuil par date sans nouvelle information prédictive ;
- les modèles de régime très dimensionnels ;
- l'ajout de profondeur ou d'arbres sans hypothèse structurée ;
- les blends dont le poids serait choisi après observation du leaderboard ;
- la sélection d'un seed favorable ;
- tout target encoding, conformément à la règle fixée.

## État au moment de la pause

- **Baseline locale :** V2 à 0,525001 OOF.
- **Meilleure ancre publique connue :** LightGBM linéaire à 0,512018.
- **Ancre publique la plus simple :** V1 Ridge à 0,511829.
- **Seule piste directionnelle non rejetée :** H11, gain multi-seed moyen de
  +0,011 point, insuffisant pour devenir baseline.
- **Outil diagnostique retenu :** H6 pour la difficulté absolue des lignes.
- **Aucun nouveau modèle prêt à soumettre** à l'issue de H13.

## Si nous reprenons : trois pistes maximum

### P1 — Refaire une cible continue plus transférable

Les deux meilleures soumissions publiques apprennent la magnitude avant de
prendre le signe. La reprise la plus rationnelle serait une régression continue
fortement shrinkée sur une cible normalisée transversalement par date/groupe,
avec comparaison pré-enregistrée à V1 et LightGBM. L'objectif est une invariance
nouvelle, pas un nouvel algorithme.

### P2 — Réponse hiérarchique d'allocation, mais très basse dimension

Partir de H11 et réduire encore la représentation : quelques facteurs d'état
pré-spécifiés, partial pooling `GROUP → ALLOCATION`, puis validation multi-seed
dès le screening. Pas de banque de 30 000 interactions.

### P3 — Validation de transfert avant tout nouvel ensemble

Conserver CV répétée + erreur standard par date + panels test-sized comme trois
vues séparées. Un ensemble ne sera envisagé que si son avantage apparaît dans
les répétitions de seeds et si une règle de combinaison est gelée avant tout
nouveau score public.

## Règle de reprise

Une hypothèse à la fois, un protocole écrit avant calcul, un screening pour le
coût mais jamais pour conclure, puis huit folds et plusieurs seeds pour tout
gain inférieur à 0,1 point. Cela évitera de repartir dans des rabbit holes.

## Addendum après la reprise H14–H16

Les trois pistes proposées ont été suivies sous gate strict :

- H14, cible continue dé-factorisée : −0,088 point au screening, arrêt ;
- H15, hiérarchie basse dimension : +0,074 point en 4-fold mais −0,020 point
  lors de la confirmation 8-fold, arrêt ;
- H16 : aucun multi-seed, panel ou ensemble lancé puisque la confirmation H15
  a échoué.

Le contrôle a donc fonctionné : aucune nouvelle baseline et aucune soumission.
Le détail est dans `42_h14_cross_sectional_target_protocol.md` à
`46_h16_transfer_gate_audit.md`.
