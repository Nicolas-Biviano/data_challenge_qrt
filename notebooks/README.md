# Notebooks à présenter

Ce dossier constitue le parcours court du projet. Les notebooks sont
pré-exécutés et se lisent dans l'ordre suivant :

1. `00_eda.ipynb` — structure des données, signal naïf, stabilité par date et
   allocation, et différences train/test ;
2. `01_baseline_v2.ipynb` — modèle logistique V2 retenu, validation groupée par
   dates, stabilité et importance des variables ;
3. `02_research_summary.ipynb` — synthèse des hypothèses testées, résultats
   robustes, pistes rejetées et limites de la validation locale.

## Lecture et réexécution

Les trois notebooks sont enregistrés avec leurs sorties et peuvent donc être
lus directement sur GitHub ou dans Jupyter. Ils cherchent automatiquement la
racine du dépôt et peuvent être lancés depuis la racine ou depuis ce dossier.

Une réexécution complète nécessite les CSV du challenge dans `data/`. Les deux
notebooks de modélisation utilisent également les artefacts de recherche locaux
conservés dans `gpt/outputs/`, qui ne sont pas versionnés. Le pipeline canonique
de la baseline est défini dans `src/models.py` et la validation groupée dans
`src/cross_validation.py`.

Les anciens notebooks de travail ne font pas partie du parcours présenté. Ils
sont conservés localement sous `archive/notebooks/`, dossier ignoré par Git.
