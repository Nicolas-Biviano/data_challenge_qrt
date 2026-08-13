"""Backward-compatible imports for :mod:`src.schema`.

New code should import ``Cols`` from ``src.schema``.
"""

from .schema import Cols


__all__ = ["Cols"]
