from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Ontology(Protocol):
    def resolve(self, value: str) -> str | None:
        """Map a free-text value to a canonical identifier (e.g. QID)."""
        ...

    def get_parents(self, canonical: str) -> list[str]:
        """Return parent canonical identifiers for hierarchy traversal."""
        ...

    def label_for(self, canonical: str) -> str | None:
        """Return a human-readable label for *canonical*, or None."""
        ...
