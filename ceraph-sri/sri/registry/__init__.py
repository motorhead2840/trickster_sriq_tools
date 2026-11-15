"""Conjecture registry for the foundation sprint."""
from __future__ import annotations

from sri.registry.conjectures import (
    VALID_STATUSES,
    Conjecture,
    get,
    list_conjectures,
    record_result,
    registry_as_dicts,
    reset_registry,
)

__all__ = [
    "VALID_STATUSES",
    "Conjecture",
    "get",
    "list_conjectures",
    "record_result",
    "registry_as_dicts",
    "reset_registry",
]
