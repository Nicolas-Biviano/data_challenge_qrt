# Presentation notebooks

This directory contains the short, reproducible presentation path for the
project:

1. `00_eda.ipynb` — data integrity, panel coverage, missing values, target
   heterogeneity, short-lag dynamics, conditional signal and train/test drift;
2. `01_retained_model.ipynb` — the retained sparse logistic pipeline,
   date-grouped validation, stability and feature contributions;
3. `02_qrt_baseline_vs_first_model.ipynb` — a paired comparison between the
   Ridge benchmark supplied by QRT and our first Ridge model.

## Reproduction

The notebooks are stored with their outputs and can be read directly on GitHub
or in Jupyter. They locate the repository root automatically, so they can be
executed from the root directory or from `notebooks/`.

A complete execution requires the challenge CSV files in `data/`. Model
preprocessing is defined in `src/models.py` and date-grouped validation in
`src/cross_validation.py`.

Earlier exploratory notebooks and the original QRT submission notebook are
kept locally under `archive/notebooks/`, which is excluded from Git.
