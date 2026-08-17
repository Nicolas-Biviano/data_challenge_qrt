# H3 — dates unanimement difficiles ou faciles

## Hypothèse

Les dates sur lesquelles cinq modèles échouent ensemble pourraient posséder un
profil de marché observable absent des dates où ils réussissent tous.

## Protocole

L'analyse porte uniquement sur les dates complètes de 276 lignes. Une date est
« difficile » si les cinq modèles sont sous 50 %, « facile » s'ils sont tous
au-dessus de 50 %, et « mixte » sinon. Un LightGBM régularisé prédit le groupe
à partir de résumés X-only, avec validation hors échantillon par dates.

## Résultats

- 300 dates difficiles ;
- 576 dates faciles ;
- 397 dates mixtes ;
- AUC H3 : 0,468791 ;
- balanced accuracy : 0,495764.

Les écarts descriptifs sont faibles : le plus grand vaut environ 0,21
écart-type. Les importances LightGBM ne se traduisent pas en pouvoir prédictif
hors échantillon.

## Décision

H3 n'est pas soutenue par les résumés actuels. Les moyennes, dispersions, parts
positives, valeurs manquantes et mesures de turnover ne permettent pas de
reconnaître les dates difficiles à l'avance.

Notebook : `gpt/notebooks/H3_hard_vs_easy_dates.ipynb`.
