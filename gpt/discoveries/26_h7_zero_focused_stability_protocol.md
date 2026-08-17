# H7 — transformations centrées sur zéro et stability selection

## Hypothèse

Les modèles distinguent bien les observations aux scores extrêmes, mais la zone
proche de zéro reste presque aléatoire. Des transformations qui dilatent les
petites variations des returns peuvent contenir un signal résiduel. Leur grand
nombre et leur corrélation imposent une sélection stable plutôt qu'un ajout en
bloc.

## Candidats pré-enregistrés

Pour chacun des huit returns de V2 (`1, 2, 3, 4, 7, 8, 9, 18`), on calcule un
z-score robuste à l'intérieur de la date, puis cinq formes :

- `signed_sqrt = sign(z) sqrt(|z|)` ;
- `tanh025 = tanh(z / 0.25)` ;
- `tanh050 = tanh(z / 0.50)` ;
- `softsign025 = z / (0.25 + |z|)` ;
- `clip025 = clip(z / 0.25, -1, 1)`.

Cela donne 40 candidats atomiques. Les huit returns bruts, les effets fixes
`ALLOCATION`/`GROUP` et les pentes `ALLOCATION × RET_1` restent toujours dans
le modèle de base.

## Sélection imbriquée

Dans chacun des huit folds externes de dates :

1. prendre uniquement les dates du train externe ;
2. effectuer huit sous-échantillonnages de 60 % de ces dates ;
3. ajuster une logistique Elastic Net (`C=0.0015`, `l1_ratio=0.8`) sur le modèle
   de base et les 40 candidats ;
4. retenir, à chaque répétition, les huit candidats aux plus grands
   coefficients absolus non nuls ;
5. sélectionner au plus quatre candidats présents dans au moins 50 % des
   répétitions, avec un signe identique dans au moins 75 % de leurs sélections.

Après classement par stabilité, un seul représentant peut être retenu par lag
de return afin d'éviter de réintroduire immédiatement des formes quasi
dupliquées.

Les 40 candidats sont en compétition atomique : deux formes corrélées d'un
même return ne sont pas ajoutées comme un groupe indivisible.

Le modèle externe est ensuite une logistique L2 V2 (`C=0.003`) enrichie des
candidats sélectionnés. Le fold externe n'intervient jamais dans la sélection.

## Critères

- fréquence de sélection et stabilité du signe ;
- accuracy et AUC OOF ;
- gain apparié à V2 par fold et bootstrap de dates ;
- accuracy dans la moitié de lignes ayant le plus faible `|score V2|` ;
- stabilité de la sélection entre folds externes.

H7 ne passe que si le gain est positif dans au moins 6 folds sur 8 et si la
borne basse du bootstrap 95 % du gain global est positive. Aucune soumission
n'est créée automatiquement.
