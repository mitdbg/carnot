from abc import ABC, abstractmethod


class Retriever(ABC):
    """
    Given a question, retrieve a list of items (pages) that are relevant to answering the question.
    """
    
    @abstractmethod
    def retrieve(self, question: str) -> list[str]:
        """
        Retrieve a list of items (pages) that are relevant to answering the question.
        """
        pass
