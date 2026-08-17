# 10 - Moyennes conditionnelles par quantile

## Pourquoi regarder les quantiles

Une correlation ou un coefficient lineaire ne voit qu'une tendance moyenne.
Les profils par quantile peuvent faire apparaitre :

- un seuil dans la queue de distribution ;
- une relation en U ;
- une saturation ;
- une interaction entre une variable non-return et `RET_1` ;
- un signal porte uniquement par les valeurs extremes.

L'audit utilise le residu de classification OOF de la V2 :

```text
residu = target_binarized - probabilite_OOF_V2
```

Une moyenne residuelle non nulle dans un quantile indique donc ce que la V2
sous-estime ou surestime encore. Les moyennes de cible ne sont jamais utilisees
comme features : ce n'est pas du target encoding.

## Controle de stabilite

Pour chacune des 47 variables non-return, le script
`gpt/conditional_quantile_audit.py` calcule :

- 10 quantiles sur les valeurs observees ;
- taux positif, cible continue moyenne, probabilite V2 et residu moyen ;
- erreur standard du residu dans chaque quantile ;
- correlation de la forme entre chacun des huit folds et la forme globale ;
- accord de signe entre folds ;
- grilles quintile `RET_1` x quintile de la feature.

Le score de classement `stable_amplitude` est l'amplitude du profil residuel
multipliee par sa correlation moyenne entre folds. Il sert a prioriser les
formes a inspecter, pas a construire une prediction.

## Relations principales

### Dispersion de `SIGNED_VOLUME_1` par date

`date_std_SIGNED_VOLUME_1` arrive en tete de l'audit :

- amplitude residuelle entre deciles : 0.03453 ;
- correlation moyenne de la forme entre folds : 0.5939 ;
- accord moyen de signe : 0.6875.

La relation est fortement non monotone. Par exemple, les deciles 7, 8 et 9 ont
des residus moyens respectifs de -0.01971, +0.01482 et -0.01391. Une colonne
lineaire brute ne peut pas representer cette alternance.

### Rang de turnover dans `TS x GROUP`

`group_date_rank_MEDIAN_DAILY_TURNOVER` montre une forme presque monotone :

- residu du premier decile : -0.00682 ;
- residu du dernier decile : +0.01116 ;
- Spearman entre decile et residu : 0.9515 ;
- correlation moyenne entre folds : 0.6315.

La grille avec `RET_1` indique que le dernier quintile de turnover est surtout
positif lorsque `RET_1` est lui-meme eleve. Cette observation a motive des
termes charniere et des interactions avec `RET_1`.

### Part de volumes positifs

`sv_positive_share_20` presente une forme courbe : residu positif dans la zone
centrale, puis negatif au-dessus d'environ 80 %. Une base quadratique, une
charniere a 80 % et un indicateur « tous positifs » ont ete testes.

### Valeur individuelle de `SIGNED_VOLUME_1`

Le profil confirme une relation non monotone et fragile. Le residu est positif
dans les deciles intermediaires, mais proche de zero dans les deux deciles
superieurs. Cela explique pourquoi valeur brute, signe et log signe n'ont pas
ameliore le modele deux-parties.

## Test des formes suggerees

Le script `gpt/quantile_shape_tests.py` transforme uniquement `X` :

- rangs compris entre 0 et 1 ;
- fonctions charnieres a 20, 40, 60 et 80 % ;
- terme quadratique pour la part de volumes positifs ;
- interaction `RET_1 x rang de turnover` ;
- aucun taux de cible par quantile n'est injecte.

### Screening sur deux folds

| Variante | Accuracy | Gain vs V2 memes lignes | Score penalise TS |
|---|---:|---:|---:|
| Part de volumes positifs, forme non lineaire | **0.525618** | +0.000506 | **0.522339** |
| Dispersion SV1 par date, spline | **0.525618** | +0.000506 | 0.522187 |
| Combinaison de toutes les formes | 0.525422 | +0.000310 | 0.522099 |
| V2 | 0.525112 | 0 | 0.521739 |
| Rang turnover lineaire | 0.524606 | -0.000506 | 0.521255 |
| Rang turnover + seuils + interaction | 0.524220 | -0.000892 | 0.520914 |

Le turnover illustre une precaution importante : une belle moyenne
conditionnelle ne garantit pas un gain predictif. Son signal est probablement
deja explique par les effets fixes, instable conditionnellement aux autres
variables, ou trop faible par rapport au bruit individuel.

### Validation huit folds

Seules les deux variantes gagnantes ont ete conservees, avec `C=0.001` et
`C=0.003`.

| Variante | C | Accuracy | Score penalise TS | Decision |
|---|---:|---:|---:|---|
| Part positive non lineaire | 0.003 | **0.525024** | **0.523325** | Gain negligeable |
| Dispersion date SV1 | 0.003 | 0.524895 | 0.523173 | Rejetee |
| Part positive non lineaire | 0.001 | 0.524766 | 0.523024 | Rejetee |
| Dispersion date SV1 | 0.001 | 0.524305 | 0.522536 | Rejetee |
| V2 | 0.003 | 0.525001 | 0.523285 | Reference |

La meilleure variante ne gagne que 0.000023 d'accuracy et 0.000040 de score
penalise TS. Elle progresse sur cinq folds et recule sur trois, avec des pertes
plus grandes que plusieurs gains. Ce niveau est indistinguable du bruit de
selection et ne justifie pas une nouvelle soumission principale.

## Conclusion

Oui, les moyennes conditionnelles par quantile aident beaucoup a comprendre les
relations. Ici elles ont :

1. confirme que la valeur brute de `SIGNED_VOLUME_1` n'a pas une pente simple ;
2. revele un regime de dispersion par date ;
3. montre une interaction plausible entre turnover et `RET_1` ;
4. produit deux transformations gagnantes au screening.

Mais la validation huit folds a aussi montre leur limite : les formes visibles
restent trop faibles ou trop variables pour ameliorer solidement la V2. Les
quantiles sont donc un excellent outil de diagnostic, pas une garantie de gain.

## Artefacts

- `gpt/conditional_quantile_audit.py` ;
- `gpt/outputs/conditional_quantile_audit/feature_summary.csv` ;
- `gpt/outputs/conditional_quantile_audit/quantile_profiles.csv` ;
- `gpt/outputs/conditional_quantile_audit/ret1_interactions.csv` ;
- `gpt/quantile_shape_tests.py` ;
- `gpt/outputs/quantile_shape_screen/` ;
- `gpt/outputs/quantile_shape_full8/`.
