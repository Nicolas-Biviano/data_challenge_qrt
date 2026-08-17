# QRT Data Challenge

This repository contains the minimal reproducible analysis and modelling path
for the QRT data challenge. The presentation is organized around three executed
notebooks:

1. `notebooks/00_eda.ipynb` establishes the data and validation constraints;
2. `notebooks/01_retained_model.ipynb` presents the retained leak-safe logistic
   pipeline;
3. `notebooks/02_qrt_baseline_vs_first_model.ipynb` compares the supplied QRT
   Ridge benchmark with the first internal Ridge model on identical folds.

## Project structure

```text
.
├── notebooks/          # Executed presentation notebooks
├── src/
│   ├── feature_engineering/  # Reusable sklearn feature transformers
│   ├── preprocessing.py      # Column selection and preprocessing graphs
│   ├── models.py             # Estimator factories only
│   ├── pipelines.py          # End-to-end modelling recipes
│   ├── cross_validation.py   # Date-grouped OOF orchestration
│   └── metrics.py            # Evaluation and paired comparisons
├── tests/              # Deterministic unit and integration checks
├── CONTRIBUTING.md     # Code documentation and validation rules
└── data/               # Local challenge data; excluded from Git
```

The local `archive/` directory is excluded from Git and is not part of the
supported project API.

## Reproduction

Place the challenge CSV files in `data/`, then run:

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
  notebooks/00_eda.ipynb \
  notebooks/01_retained_model.ipynb \
  notebooks/02_qrt_baseline_vs_first_model.ipynb
```

Validation splits complete `TS` groups rather than individual rows. Complete
scikit-learn pipelines are cloned and fitted independently inside each fold so
that learned preprocessing cannot leak across the validation boundary. The
validation layer receives raw frames by default; each pipeline owns its column
selection and feature engineering.
