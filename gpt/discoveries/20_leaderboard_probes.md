# Probes leaderboard après l'échec public de V2

## Information publique disponible

| Modèle | Accuracy OOF | Score public rapporté |
|---|---:|---:|
| V1 Ridge effets fixes | 0,524709 | environ 0,5118 |
| V2 logistique effets fixes | 0,525001 | environ 0,5070 |

Le leaderboard inverse donc le classement V1/V2 donné par la CV aléatoire par
dates. V2 n'est plus considéré comme modèle final : il reste uniquement une
baseline OOF.

## Correction sur l'adversarial validation

L'expérience avait déjà été exécutée le 2 août. Le discriminateur train/test
atteint une AUC de 0,973 avec les profils de date, ce qui prouve un fort
covariate shift observable. Mais V2 atteint 0,5336 sur un holdout frais composé
des 120 dates train les plus proches du test. La sélection X-only de dates
test-like ne reproduit donc pas la chute publique à 0,507.

Cela suggère soit un changement de `P(y|X)` non détectable avec X seul, soit un
public leaderboard très particulier. Avec seulement deux scores publics, il
est impossible de départager ces hypothèses.

## Trois soumissions diagnostiques

| Priorité | Probe | Accuracy OOF | Taux positif test | Question testée |
|---:|---|---:|---:|---|
| 1 | LightGBM feuilles linéaires, returns compacts | 0,524442 | 0,55064 | un modèle non linéaire et plus divers transfère-t-il mieux ? |
| 2 | Ridge sur huit returns uniquement | 0,520404 | 0,56545 | les effets `ALLOCATION/GROUP` causent-ils le mauvais transfert ? |
| 3 | `sign(RET_1)` | 0,518924 | 0,51519 | le signal le plus simple survit-il au shift ? |

Après réception de l'historique public complet, un quatrième fichier a été
ajouté : blend 50/50 des scores continus de V1 et du LightGBM linéaire. Les deux
modèles sont les meilleurs publics connus et désaccordent sur 16,0 % des lignes
OOF. Sans tuning du poids, le mélange 50/50 atteint 0,525202 OOF, contre
0,524709 pour V1 et 0,524442 pour LightGBM.

Le score public du blend est **0,510700** : il ne dépasse aucun composant. Les
autres poids ne seront pas explorés sur le leaderboard.

Le LightGBM désaccorde avec V2 sur 18,35 % des lignes test. Le Ridge
returns-only désaccorde avec V2 sur 18,26 %, et `sign(RET_1)` sur 24,36 %.
Ces probes sont donc beaucoup plus informatives que de nouvelles variantes
logistiques quasi identiques.

## Ordre conseillé

Soumettre d'abord LightGBM. S'il dépasse V1, le prochain axe doit être la
non-linéarité et la diversité. S'il échoue mais que le Ridge returns-only bat
V1, les effets catégoriels sont probablement fragiles. Si `sign(RET_1)` bat les
modèles entraînés, le problème est plus profond : notre apprentissage exploite
principalement des relations propres au train.

Les fichiers sont dans `gpt/outputs/leaderboard_probes/`. Le script
reproductible est `gpt/build_leaderboard_probes.py`.
