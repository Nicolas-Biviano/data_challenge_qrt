# Journal de recherche - modele GPT V2

Ce journal suit les experiences dans l'ordre ou elles sont realisees. Les
details, formules et tableaux complets sont ranges dans `gpt/discoveries/`.

## Regles de recherche

1. Les labels `DATE_xxxx` ne sont jamais interpretes comme un ordre temporel.
2. Les folds sont construits sur les valeurs uniques de `TS`: une date complete
   appartient soit au train, soit a la validation.
3. Les transformations apprises (imputation, normalisation, encodage) sont
   ajustees sur le fold d'entrainement uniquement.
4. Les agregats transversaux d'une date utilisent uniquement les variables de
   `X`, jamais `target`.
5. La reconstruction de dates par recouvrement des fenetres est hors perimetre.
6. Aucun target encoding n'est autorise, y compris pour `ALLOCATION` ou
   `GROUP`. Toute representation categorielle doit etre independante de `y`.
7. Une feature n'entre dans la V2 que si son gain est suffisamment stable entre
   les folds et ne repose pas sur un seul sous-groupe.

## Mesures principales

- accuracy OOF globale;
- accuracy moins une erreur standard calculee par `TS`;
- stabilite par fold, `TS`, `ALLOCATION` et `GROUP`;
- comparaison sur des folds fixes (`n_splits=8`, `random_state=0`).

## Experiences

| ID | Experience | Accuracy OOF | Score penalise TS | Decision |
|---|---|---:|---:|---|
| V1 | Ridge, effets fixes allocation/groupe, pentes allocation x RET_1 | 0.524709 | 0.522975 | Baseline officielle |
| F01 | Les 20 rendements bruts | 0.523616 | 0.521898 | Rejete |
| F02 | Rendements et dynamique sur 20 jours | 0.523315 | 0.521619 | Rejete |
| F03 | Baseline et liquidite resumee | 0.524550 | 0.522823 | Neutre, a cibler |
| F04 | Baseline et 20 volumes bruts | 0.524185 | 0.522457 | Rejete |
| F05 | Regime transversal par date | 0.523356 | 0.521638 | Rejete dans cette forme lineaire |
| F06 | Regime transversal date-groupe | 0.522632 | 0.520910 | Rejete dans cette forme lineaire |
| F07 | Toutes les familles combinees | 0.520605 | 0.518878 | Rejete |
| C01 | Grille Ridge alpha 10 a 1000 | 0.524709 | 0.522975 | alpha=100 confirme |
| C02 | Penalisation relative allocation x RET_1 | 0.524709 | 0.522975 | echelle 1 conservee |
| C03 | Seuil global calibre hors fold | 0.524140 | non retenu | Rejete |
| M01 | HistGradientBoosting regression, screening 2 folds | 0.519876 | 0.516376 | Rejete |
| M02 | HistGradientBoosting classification, screening 2 folds | 0.522656 | 0.519268 | Rejete |
| M03 | Logistique sparse C=0.003 | 0.525001 | 0.523285 | Candidat, gain faible |
| R01 | Regime date seul, meilleur modele | 0.506713 | 0.504802 | Rejete seul |
| R02 | Regime date-groupe ExtraTrees | 0.511774 | 0.509916 | Rejete seul, candidat ensemble |
| S01 | Ridge + dispersion groupe de SV1 | 0.524935 | 0.523201 | Gain fragile, non retenu |
| S02 | Ridge + count date SV1 | 0.525045 | 0.523303 | Gain lie au missingness, non retenu |
| S03 | Ridge + count SV1 + dispersion SV2 | 0.525090 | 0.523357 | Non retenu pour limiter le data artifact |
| E01 | Ensemble Ridge + logistique, poids hors fold | 0.524426 | 0.522706 | Rejete |
| V2 | Logistique sparse C=0.003 | 0.525001 | 0.523285 | Modele final |
| V3-L01 | LightGBM constant, returns compacts, 2 folds | 0.524190 | 0.520959 | Rejete au screening |
| V3-L02 | LightGBM linear tree, returns compacts, 2 folds | 0.525014 | 0.521876 | Sous V2 sur les memes lignes |
| V3-L04 | LightGBM linear tree, returns compacts, 8 folds | 0.524442 | 0.522852 | Rejete apres validation complete |
| V3-L03 | LightGBM constant, combined, 2 folds | 0.523994 | 0.520634 | Rejete au screening |
| V3-N01 | Non-return + GROUP, 2 folds | 0.512252 | 0.508695 | Signal faible |
| V3-N02 | Non-return + GROUP + cluster X-only, 2 folds | 0.514708 | 0.511172 | Cluster utile mais insuffisant |
| V3-N03 | Non-return + categories completes, 2 folds | 0.515682 | 0.512606 | Rejete seul |
| V3-Q01 | QRF moyenne, 1 fold, 20 arbres | 0.513228 | 0.508887 | Rejete au screening |
| V3-Q02 | QRF mediane, 1 fold, 20 arbres | 0.510228 | 0.505843 | Rejete au screening |
| V3-D01 | LightGBM lineaire, selection imbriquee profondeur/regularisation, 2 folds externes | 0.524835 | non calcule | Overfit corrige, mais sous V2 et profondeur instable |
| V3-V01 | Volume 1 global deux-parties, meilleur screening 2 folds | 0.525059 | 0.521744 | Sous la V2 sur les memes lignes |
| V3-V02 | Specialiste si volume 1 observe, fallback V2, 8 folds | 0.525185 | 0.523483 | Petit gain global, mais seulement 4 folds sur 8 |
| V3-V03 | Shrinkage V2-specialiste choisi hors fold | 0.524574 | non calcule | Rejete |
| V3-QM01 | Part de volumes positifs, forme non lineaire, 8 folds | 0.525024 | 0.523325 | Gain trop faible pour remplacer V2 |
| V3-QM02 | Dispersion SV1 par date, spline, 8 folds | 0.524895 | 0.523173 | Rejetee |
| AUD-P01 | Correlation des erreurs et shift train-test | - | - | Public ~0.50: CV aleatoire par dates a remettre en question |
| H1 | Entraînement sur dates complètes uniquement | 0.518318 | comparaison appariée | Sous toutes les dates, rejeté |
| H2 | Seuils appris par profils de marché | 0.521865 à 0.523144 | bootstrap TS | Sous le seuil 0,5, rejeté |
| H3 | Classification dates difficiles vs faciles | AUC 0.468791 | balanced accuracy 0.495764 | Indiscernable du hasard |
| H3-L | Détection X-only des lignes unanimement fausses | AUC 0.523875 | 5 folds par TS | Faible et opposé selon y |
| H4-A | Base non linéaire globale | 0.524618 | dates complètes 0.518056 | Gate échoué, H4-B à E arrêtées |
| H4-F | Forward imbriquée par groupes | 0.525233 | gain IC95 % [-0.000670 ; 0.001127] | Gain non validé |
| H4-G | Forward imbriquée atomique et résiduelle | 0.524686 | gain IC95 % [-0.001108 ; 0.000423] | Rejetée ; formes atomiques instables |
| LB-P | Trois probes leaderboard diversifiées | 0.518924 à 0.524442 | public à renseigner | LightGBM, Ridge returns-only, sign(RET_1) |
| LB-A | Alignement CV/public sur trois runs reliés | Spearman -1.0 | stress/public -0.5 | Aucune validation locale ne classe correctement les trois |
| H5-P | 1 000 panels de 120 dates appariés X-only | classement public exact 25,3 % vs 14,9 % uniforme | P(LightGBM > V1) 57,6 % | Meilleur stress test, trop incertain pour tuner |
| H5-C | Confiance par décile de `|score|` | V1 50,38 % à 57,24 % | pente positive 8/8 folds | Confiance absolue validée, sélecteur de modèle rejeté |
| H6-E | Probabilité d'erreur compacte cross-fittée | AUC 0,5235 ; Brier 0,24885 | bat constant 8/8 folds | Retenue pour difficulté/diagnostic |
| H6-S | Sélection V1/LightGBM par erreur estimée | meilleur OOF 0,525280 | 6/8 folds, 65,2 % panels | Gate échoué, aucune soumission |
| H7 | Stability selection de 40 formes zero-focused | 0,524519 | gain IC95 % [-0,001197 ; 0,000209] | Stable mais redondant, gate échoué |
| H8-R | Reproductibilité des pentes par date | split-half 0,757 à 0,793 | accord de signe 77,8 à 80,4 % | Inversions réelles |
| H8-P | Prédiction X-only des inversions | meilleure AUC 0,5283 | tous R² négatifs | Profil marginal insuffisant, gate échoué |
| H9-A | Archétypes comportementaux X-only | ARI folds 0,878 | AMI avec GROUP 0,126 | Segmentation stable et nouvelle |
| H9-R | Profil relationnel de 290 variables | meilleure AUC OOF 0,5435 | AUC train ~0,70 ; MAE perd 63/64 | Overfit, gate échoué |
| H10 | Returns dé-factorisés marché/GROUP, leave-one-out | meilleur 0,523666 vs V2 0,525001 | résidus seuls perdent 8/8 folds | Rejeté, PCA non lancée |
| H11 | `ALLOCATION ×` huit états X-only, shrinkage fort | 0,525318 vs V2 0,525001 | +0,032 point, 6/8 folds, IC contient 0 | Candidat probe, pas nouvelle baseline |
| H11-MS | H11 gelé, seeds CV 0 à 4 | gain moyen +0,000109 | 4/5 seeds, 25/40 folds, IC contient 0 | Signal très faible |
| H12-FM | Correction bilinéaire low-rank rang 4 | 0,519385 vs V2 0,525112, 2 folds | gap train-valid 0,0163 | Rejetée |
| H12-LGBM | LightGBM conditionnel profondeur 3 | linéaire 0,524072, constant 0,523723 | profondeur réelle max 3, gap >0,010 | Rejeté |
| H13-RF | Random Forest tunée imbriquée, 1 000 arbres très régularisés | 0,523711 vs V2 0,525001 | gain −0,001290, SE date 0,001097, 4/8 folds | Rejetée ; convergence acquise, pas d'overfit massif |
| H14 | Ridge sur cible continue dé-factorisée date/groupe, screening 4 folds | meilleur 0,525103 vs brut 0,525981 | AUC légèrement supérieure, 0/4 folds gagnés | Rejeté au screening |
| H15 | Réponse hiérarchique 3 états, shrinkage groupe/allocation | 4-fold +0,000738 puis 8-fold −0,000195 | 3/8 folds, IC95 gain [-0,001147 ; 0,000756] | Rejeté à la confirmation |
| H16 | Gate transfert avant ensemble | multi-seed/panels non lancés | H14 et H15 n'ont pas passé leurs gates | Aucun ensemble, aucune soumission |

## Prochaine etape

Le diagnostic complet de regularisation est dans
`discoveries/08_lightgbm_bias_variance_diagnostic.md`. Il confirme que le
LightGBM courant sur-apprend, mais la selection imbriquee fortement regularisee
reste sous V2 sur deux folds externes. La V2 finale reste la logistique sparse
`C=0.003`. L'etude deux-parties de `SIGNED_VOLUME_1` est documentee dans
`discoveries/09_signed_volume_two_part.md`; son petit gain moyen n'est pas assez
stable pour remplacer V2. L'audit des moyennes conditionnelles et ses tests de
formes non lineaires sont dans `discoveries/10_conditional_quantile_means.md`.
La chute du score public, le biais de predictions positives et la correlation
des erreurs sont analyses dans
`discoveries/11_public_score_and_error_correlation.md`.
Les trois hypothèses suivantes sont isolées dans
`discoveries/12_h1_test_like_training.md`,
`discoveries/13_h2_auc_and_thresholds.md` et
`discoveries/14_h3_hard_vs_easy_dates.md`, avec leurs notebooks dédiés dans
`gpt/notebooks/`.
L'analyse H3 au niveau ligne, les distances de distribution et le diagnostic
des transformations non linéaires sont dans
`discoveries/15_h3_row_error_distributions.md`.
Les nouvelles transformations sont isolées dans H4 et décrites dans
`discoveries/16_h4_nonlinear_group_features.md`.
Les résultats et la décision d'arrêt sont dans
`discoveries/17_h4_execution_results.md`.
Le protocole atomique/résiduel et la comparaison H4-F/H4-G sont dans
`discoveries/18_h4_atomic_residual_protocol.md` et
`discoveries/19_h4_nested_selection_results.md`.
Les probes destinées à recaler la validation sur le leaderboard sont décrites
dans `discoveries/20_leaderboard_probes.md`.
L'historique des treize soumissions et l'audit d'alignement sont dans
`discoveries/21_public_history_alignment.md`.
Le protocole H5 des panels de 120 dates est dans
`discoveries/22_test_sized_panels_protocol.md`; ses résultats et l'analyse de
confiance par amplitude sont dans
`discoveries/23_test_sized_panels_results.md` et le notebook dédié
`notebooks/H5_test_sized_panels_and_confidence.ipynb`.
Le protocole et les résultats H6 sont dans
`discoveries/24_h6_error_estimator_protocol.md` et
`discoveries/25_h6_error_estimator_results.md`, avec le notebook
`notebooks/H6_conditional_error_estimator.ipynb`.
H7 est documenté dans `discoveries/26_h7_zero_focused_stability_protocol.md`
et `discoveries/28_h7_zero_focused_stability_results.md`. H8 est documenté
dans `discoveries/27_h8_slope_inversion_protocol.md` et
`discoveries/29_h8_slope_inversion_results.md`. Leurs notebooks séparés sont
dans `gpt/notebooks/`.
Les archétypes H9 et leur tentative de prédiction relationnelle sont documentés
dans `discoveries/30_h9_allocation_archetypes_relational_protocol.md` et
`discoveries/31_h9_allocation_archetypes_relational_results.md`, avec le
notebook `notebooks/H9_allocation_archetypes_and_relational_regimes.ipynb`.
La décomposition H10 des returns en marché, écart de groupe et résidu propre est
décrite dans `discoveries/32_h10_defactored_returns_protocol.md` et ses résultats
dans `discoveries/33_h10_defactored_returns_results.md`, avec le notebook
`notebooks/H10_defactored_returns.ipynb`.
Les fonctions de réaction H11 sont décrites dans
`discoveries/34_h11_conditional_allocation_response_protocol.md` et leurs
résultats dans `discoveries/35_h11_conditional_allocation_response_results.md`,
avec le notebook `notebooks/H11_conditional_allocation_response.ipynb`.
Le protocole des modèles puissants, l'audit multi-seed et les résultats H12 sont
dans `discoveries/36_powerful_models_protocol.md`,
`discoveries/37_h11_multiseed_results.md` et
`discoveries/38_h12_powerful_models_results.md`, avec le notebook
`notebooks/H12_multiseed_and_powerful_models.ipynb`.
La Random Forest H13 tunée selon la littérature est décrite dans
`discoveries/39_h13_paper_tuned_random_forest_protocol.md` et
`discoveries/40_h13_paper_tuned_random_forest_results.md`, avec le notebook
`notebooks/H13_paper_tuned_random_forest.ipynb`.
Le bilan transversal préparé pour la pause est dans
`discoveries/41_pause_wrap_up.md`.
La reprise contrôlée H14–H16 est documentée dans
`discoveries/42_h14_cross_sectional_target_protocol.md` à
`discoveries/46_h16_transfer_gate_audit.md`.
