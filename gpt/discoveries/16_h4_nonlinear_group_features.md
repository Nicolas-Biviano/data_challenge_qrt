# H4 — nouvelles features non linéaires et interactions groupées

## Hypothèse

Une petite base de transformations saturantes sur des coordonnées relatives à
`TS` ou `TS × GROUP` peut capturer des relations manquées par les modèles
linéaires, tout en restant plus régulière qu'un arbre profond.

## Biais inductifs retenus

- `tanh(z/c)` : effet monotone saturant ;
- `arctan(z)` : saturation lente ;
- `sign(z) × log(1+|z|)` : compression douce des queues ;
- hinges : relation linéaire par morceaux ;
- `arcsin(2r−1)` : amplification des rangs extrêmes, testée séparément.

`tan(z)` est exclu à cause de ses pôles et de son biais périodique sans
justification financière.

## Variables préenregistrées

- `RET_1` et ses rangs dans `TS` et `TS × GROUP` ;
- turnover relatif à `TS` et `TS × GROUP` ;
- disponibilité de `SIGNED_VOLUME_1` ;
- part des volumes signés positifs observés.

## Étapes

1. H4-A : base non linéaire globale ;
2. H4-B : interactions `GROUP × f(z)` ;
3. H4-C : interactions ciblées `ALLOCATION × f(RET_1)` ;
4. H4-D : interactions avec la disponibilité de SV1 ;
5. H4-E : amplification des queues par `arcsin`.

Chaque étape n'est ouverte que si la précédente passe les gates.

## Validation

- huit folds de dates entières ;
- aucun ordre temporel ;
- aucun target encoding ;
- comparaison appariée à V2 ;
- bootstrap des dates à 95 % ;
- accuracy globale et dates complètes ;
- AUC, taux positif et métriques par classe ;
- pénalisation des pentes `GROUP`, puis pénalisation plus forte des pentes
  `ALLOCATION`.

## Statut

Protocole exécuté le 5 août 2026. H4-A n'a pas passé le gate ; voir
`gpt/discoveries/17_h4_execution_results.md`.

Notebook : `gpt/notebooks/H4_nonlinear_group_features.ipynb`.
