"""Provenance code that is not part of the supported API.

Importing this package warns. The public ``sri`` package does not import it.
See ``sri/legacy/README.md``.
"""
from __future__ import annotations

import warnings

warnings.warn(
    "sri.legacy is provenance code, not the supported API. See sri/legacy/README.md.",
    DeprecationWarning,
    stacklevel=2,
)
