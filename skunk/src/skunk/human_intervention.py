from __future__ import annotations

from skunk.common import HumanInterventionHandler, PendingHumanIntervention
from skunk.multi_turn_agent import Tool


class RequestHumanTool(Tool):
    name = "request_human"

    doc = """\
### request_human(kind: str, instructions: str, context: str | None = None, source_docs: list[str] | None = None)
Request human help only when the available retrieval and lookup tools cannot
obtain the needed information. Use `kind="pdf_extraction"` when a person must
inspect a PDF, table, chart, or image, and `kind="external_lookup"` when a
person must find authoritative external information. Give precise instructions,
include useful context, and list any source references already identified. The
call pauses until a human responds, then returns a dict with `response` and
`source_docs`.

```python
request_human(
    kind="pdf_extraction",
    instructions="Read the total shown in the final row of Table 4.",
    context="Treasury Bulletin 1954-02, page 17",
    source_docs=["Treasury Bulletin 1954-02 PDF page 17"],
)
```"""

    def __init__(self, handler: HumanInterventionHandler):
        self._handler = handler

    def __call__(
        self,
        kind: str,
        instructions: str,
        context: str | None = None,
        source_docs: list[str] | None = None,
    ) -> PendingHumanIntervention:
        cleaned_kind = str(kind).strip()
        cleaned_instructions = str(instructions).strip()
        if not cleaned_kind:
            raise ValueError("human intervention kind must not be empty")
        if not cleaned_instructions:
            raise ValueError("human intervention instructions must not be empty")
        cleaned_sources = [
            str(item).strip() for item in (source_docs or []) if str(item).strip()
        ]
        return PendingHumanIntervention(
            self._handler(
                cleaned_kind,
                cleaned_instructions,
                str(context).strip() if context is not None else None,
                cleaned_sources,
                None,
            )
        )
