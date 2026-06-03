from __future__ import annotations

from abc import ABC, abstractmethod

from skunk.common import ExecutionContext


class Retriever(ABC):
    """
    Given a question, retrieve a list of items (pages) that are relevant to answering the question.
    """

    @abstractmethod
    async def retrieve(
        self,
        ctx: ExecutionContext,
        question: str,
        *,
        branch_key: str | None = None,
        branch_period: str | None = None,
        branch_as_of: str | None = None,
    ) -> list[str]:
        """
        Retrieve a list of items (page keys) relevant to `question`. Optional branch
        hints (`branch_key` / `branch_period` / `branch_as_of`) narrow the search.
        """
        ...
