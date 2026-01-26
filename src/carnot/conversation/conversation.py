from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str

class Conversation:
    def __init__(self, user_id: str, session_id: str, title: str, dataset_ids: list[str], messages: list[Message] | None = None):
        self.user_id = user_id
        self.session_id = session_id
        self.title = title
        self.dataset_ids = dataset_ids
        self.messages = messages or []

    def condense(self, query: str) -> str:
        """
        Condense the conversation history relevant to the query.
        """
        return ""
