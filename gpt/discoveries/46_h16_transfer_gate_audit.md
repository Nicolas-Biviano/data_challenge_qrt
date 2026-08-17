# H16 — Audit du gate de transfert avant ensemble

## Objet

H16 devait empêcher qu'un gain de screening, un seed favorable ou un panel
test-like suffise à autoriser un ensemble. Aucun modèle H14/H15 n'ayant passé la
confirmation, l'audit s'arrête avant le multi-seed et avant les panels.

Ce n'est pas une expérience manquante : c'est l'application du gate.

## Chemin de décision

| Étape | Candidat | Résultat | Décision |
|---|---|---:|---|
| H14, 4 folds | cible `market_100` | −0,000878, 0/4 folds | arrêt immédiat |
| H15, 4 folds | hiérarchie 3 états | +0,000738 | confirmation autorisée |
| H15, 8 folds | même modèle gelé | −0,000195, 3/8 folds | rejet |
| H16 multi-seed | H15 | non lancé | gate 8-fold échoué |
| H16 panels test-sized | H15 | non lancé | aucun candidat transférable |
| Ensemble | V2 + H15 | non construit | interdit par protocole |

## Ce que ce contrôle a évité

Après le screening 4-fold, H15 aurait pu être annoncé comme un gain de
+0,074 point avec meilleure AUC. Les huit folds montrent que ce récit était
faux pour l'accuracy globale. Tester ensuite plusieurs seeds, choisir le
meilleur, apprendre un seuil ou optimiser un poids de blend aurait transformé
le bruit de validation en résultat.

## Bilan des trois directions de reprise

1. **Cible continue plus transférable : rejetée pour l'accuracy.** La
   dé-factorisation améliore parfois l'AUC, mais perd le niveau utile au signe.
2. **Hiérarchie basse dimension : signal de ranking, pas de gain d'accuracy.**
   Le screening positif ne survit pas aux huit folds.
3. **Validation avant ensemble : réussie méthodologiquement.** Elle a bloqué un
   faux positif avant multi-seed, panels et soumission.

## État final

- V2 reste la baseline locale à 0,525001.
- LightGBM linéaire reste la meilleure ancre publique connue à 0,512018.
- H11 reste directionnel mais non établi.
- H14 et H15 ne produisent aucune soumission.
- aucune mise en forme ou publication GitHub n'a été effectuée.

La prochaine phase est volontairement séparée : inventaire, architecture du
dépôt, nettoyage, documentation et publication ne commenceront qu'après accord
explicite sur chaque transformation.
