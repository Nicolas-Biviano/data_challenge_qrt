"""Backward-compatible imports for the renamed :mod:`src.data` module.

New code should import ``ChallengeDataLoader`` from ``src.data``.  This module
is intentionally kept while the research scripts are migrated.
"""

from .data import PATH_DATA, ChallengeDataLoader


__all__ = ["PATH_DATA", "ChallengeDataLoader"]
