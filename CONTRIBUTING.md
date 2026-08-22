# Code documentation standard

The repository uses English NumPy-style docstrings. Documentation should make
the data contract, validation boundaries, and leakage safeguards explicit
without repeating the implementation line by line.

## Required coverage

- Every public module starts with a one-sentence description of its scope.
- Stable modules declare their supported import surface in `__all__`.
- Every public function, method, class, and dataclass has a docstring.
- Private helpers need a docstring only when their contract or rationale is not
  evident from their name and implementation.
- Compatibility modules state the canonical import path and why the alias is
  retained.

## Docstring structure

Use the following sections only when they add information:

1. A short imperative or descriptive summary.
2. A brief paragraph for context that is not clear from the signature.
3. `Parameters` for every argument other than `self` and `cls`.
4. `Returns` for every non-`None` return value.
5. `Raises` for errors that are part of the public contract.
6. `Notes` for invariants, leakage controls, statistical interpretation, or
   backward-compatibility constraints.
7. `Examples` only when correct usage is not obvious from the signature.

Types belong in Python annotations. Docstrings describe meaning, shape,
alignment, units, and constraints rather than duplicating type annotations.

## Comments

- Explain why a decision is necessary, not what the next line does.
- Document non-obvious leakage barriers and numerical safeguards close to the
  relevant code.
- Remove stale commented-out code instead of explaining it.
- Do not narrate standard pandas, NumPy, or scikit-learn operations.

## Research workflow

New model ideas move through three tiers, in order. Each tier has a different
bar for polish and a different home in the repo.

1. **Scratch** — `research/scratch/` (gitignored). Try anything here: call
   `run_cv` directly from a throwaway notebook or script, no docstrings, no
   tests, no narrative.
2. **Tracked research** — a candidate worth remembering. Call
   `src.experiment_log.log_candidate(name, cv_result, estimator, notes=...)`
   after a `run_cv` run; it appends one row to the committed
   `research/candidates.csv` with the candidate's parameters and a narrow set
   of headline metrics. The log's git history is itself a legible trail of
   what was tried and why it was kept or rejected — that is the point of
   committing it rather than using a local-only experiment tracker.
3. **Presented** — a numbered notebook under `notebooks/` (see
   `01_retained_model.ipynb`). A candidate is promoted here once it is
   retained: rebuilt as a canonical factory in `src/models.py` /
   `src/pipelines.py`, exported via `__all__`, documented per this file, and
   narrated with the same rigor as the existing presentation notebooks.

## Project-specific contracts

- State whether frames must be indexed by `ROW_ID` and whether indices must be
  aligned.
- State whenever complete `TS` groups must remain together.
- Make clear which preprocessing is learned inside each validation fold.
- Distinguish model-selection stability diagnostics from econometric standard
  errors under panel dependence.
- Document probability thresholds and positive-class conventions explicitly.

## Maintenance rule

A code change that modifies public behavior must update its docstring in the
same commit. Documentation-only changes must not alter runtime behavior.

Before committing a documentation change, run:

```bash
.venv/bin/python -m pydocstyle \
  src/__init__.py src/data.py src/schema.py \
  src/cross_validation.py src/metrics.py src/models.py src/preprocessing.py \
  src/pipelines.py src/experiment_log.py src/feature_engineering/__init__.py \
  src/feature_engineering/interactions.py
.venv/bin/python -m pytest tests -q
```