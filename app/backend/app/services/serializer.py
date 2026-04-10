"""Sanitise arbitrary Python objects for storage in PostgreSQL JSONB columns.

PostgreSQL imposes constraints on text stored in ``jsonb`` columns that
standard :func:`json.dumps` does not enforce (e.g. null bytes ``\\u0000``
are forbidden).  This module centralises all sanitisation rules in a
single, extensible class so that every write path goes through the same
cleaning pipeline.

Representation invariant:
    The ``_text_cleaners`` list is non-empty and each element is a
    ``(pattern: str, replacement: str)`` pair applied in order.

Abstraction function:
    Represents a pipeline that converts an arbitrary Python object into
    a *jsonb-safe* Python value (``dict``, ``list``, scalar) that can be
    assigned directly to a SQLAlchemy JSONB column without triggering
    PostgreSQL encoding errors.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

logger = logging.getLogger("uvicorn.error")


class JsonbSerializer:
    """Convert arbitrary Python values to PostgreSQL-safe JSONB payloads.

    The two-phase pipeline is:

    1. **Type coercion** — ``json.dumps(value, default=_default_encoder)``
       converts non-JSON-native types (``datetime``, ``Decimal``,
       ``bytes``, custom objects) to JSON-native primitives.
    2. **Text sanitisation** — the resulting JSON *string* is scrubbed of
       characters that PostgreSQL rejects (e.g. null bytes).

    The result is re-parsed via ``json.loads`` so the caller receives a
    clean ``dict``/``list``/scalar ready for SQLAlchemy column assignment.

    Representation invariant:
        - ``_text_cleaners`` is a non-empty list of
          ``(target: str, replacement: str)`` pairs.

    Abstraction function:
        Represents a reusable serialisation pipeline from arbitrary
        Python values to PostgreSQL-jsonb-safe Python values.

    Usage::

        ser = JsonbSerializer()
        safe = ser.sanitize(preview_dict)   # safe to assign to a JSONB column
        safe = ser.sanitize(cells_list)     # works on lists too
    """

    # Characters / sequences that PostgreSQL's text/jsonb storage rejects.
    # Each entry is (target_substring, replacement).
    #
    # Note: json.dumps encodes a raw null byte (0x00) as the six-character
    # escape sequence \\u0000 in the output string.  We must strip BOTH the
    # raw byte AND the JSON escape form, because the data may contain either
    # depending on how it was produced (PDF extractors emit raw 0x00, while
    # some serialisation paths may produce the literal text \u0000).
    _text_cleaners: list[tuple[str, str]] = [
        ("\\u0000", ""),  # JSON-escaped null byte (literal 6-char sequence in the JSON string)
        ("\x00", ""),     # Raw null byte (single character, just in case)
    ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sanitize(self, value: Any) -> Any:
        """Convert *value* to a JSONB-safe Python object.

        Requires:
            None — any Python value is accepted.

        Returns:
            A JSON-native Python object (``dict``, ``list``, ``str``,
            ``int``, ``float``, ``bool``, or ``None``) that can be
            stored in a PostgreSQL ``JSONB`` column without encoding
            errors.

        Raises:
            ``ValueError`` if *value* cannot be serialised to JSON
            even with the fallback encoder.
        """
        json_str = json.dumps(value, default=self._default_encoder)
        json_str = self._clean_text(json_str)
        return json.loads(json_str)

    # ------------------------------------------------------------------
    # Extension points
    # ------------------------------------------------------------------

    @staticmethod
    def _default_encoder(obj: Any) -> Any:
        """Fallback encoder for :func:`json.dumps`.

        Handles types that the standard JSON encoder cannot serialise.
        Extend this method (via subclassing) to add first-class support
        for new media types such as images, plots, or audio.

        Requires:
            ``obj`` is a value that :func:`json.dumps` could not
            serialise with the default encoder.

        Returns:
            A JSON-serialisable representation of *obj*.

        Raises:
            ``TypeError`` if *obj* cannot be converted.
        """
        # bytes / bytearray → base64 string (future: images, audio)
        if isinstance(obj, (bytes, bytearray)):
            return base64.b64encode(obj).decode("ascii")

        # Anything with a ``model_dump`` method (Pydantic models)
        if hasattr(obj, "model_dump"):
            return obj.model_dump()

        # Anything with a ``to_dict`` / ``__dict__`` method
        if hasattr(obj, "to_dict"):
            return obj.to_dict()

        # Last resort — ``str(obj)`` matches the previous ``default=str``
        # behaviour and keeps datetime, Decimal, etc. readable.
        return str(obj)

    def _clean_text(self, text: str) -> str:
        """Apply all registered text cleaners to *text*.

        Requires:
            ``text`` is a valid JSON string.

        Returns:
            A cleaned JSON string with all problematic sequences
            removed or replaced.

        Raises:
            None.
        """
        for target, replacement in self._text_cleaners:
            text = text.replace(target, replacement)
        return text


# Module-level singleton so callers don't need to instantiate.
jsonb_serializer = JsonbSerializer()
