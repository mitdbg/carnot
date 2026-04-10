"""Integration tests for the ``/api/conversations`` routes.

Validates conversation CRUD operations against the live Docker Compose
stack: create, list, detail, add message, update title, and delete.

All tests require ``RUN_INTEGRATION_TESTS=1`` and valid Auth0 credentials.
"""

from __future__ import annotations

import uuid

import requests


def _make_session_id() -> str:
    """Return a unique session ID for a new conversation."""
    return f"integ-test-{uuid.uuid4().hex[:12]}"


class TestConversationsAPI:
    """CRUD operations on the ``/api/conversations`` endpoints.

    Representation invariant:
        Every conversation created during a test is registered in
        ``created_conversation_ids`` so the fixture teardown can delete it.

    Abstraction function:
        Represents the contract between the frontend chat UI and the
        FastAPI conversation routes, exercised through
        nginx → uvicorn → PostgreSQL.
    """

    def test_create_conversation(
        self,
        backend_url: str,
        auth_headers: dict,
        created_conversation_ids: list[int],
    ) -> None:
        """``POST /api/conversations/`` creates a new conversation.

        Returns:
            200 with ``ConversationResponse`` containing ``id``,
            ``session_id``, ``title``, ``message_count == 0``.
        """
        session_id = _make_session_id()
        resp = requests.post(
            f"{backend_url}/conversations/",
            headers=auth_headers,
            json={"session_id": session_id, "title": "Integration Test Conv"},
            timeout=10,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["session_id"] == session_id
        assert body["title"] == "Integration Test Conv"
        assert body["message_count"] == 0
        assert "id" in body
        created_conversation_ids.append(body["id"])

    def test_list_conversations(
        self,
        backend_url: str,
        auth_headers: dict,
        created_conversation_ids: list[int],
    ) -> None:
        """``GET /api/conversations/`` includes a conversation we created.

        Returns:
            200 with a list containing the created conversation.
        """
        session_id = _make_session_id()
        create_resp = requests.post(
            f"{backend_url}/conversations/",
            headers=auth_headers,
            json={"session_id": session_id, "title": "List Test Conv"},
            timeout=10,
        )
        assert create_resp.status_code == 200, create_resp.text
        conv_id = create_resp.json()["id"]
        created_conversation_ids.append(conv_id)

        # List
        resp = requests.get(
            f"{backend_url}/conversations/",
            headers=auth_headers,
            timeout=10,
        )
        assert resp.status_code == 200, resp.text
        conversations = resp.json()
        match = [c for c in conversations if c["id"] == conv_id]
        assert len(match) == 1, f"Conversation {conv_id} not in list"
        assert match[0]["title"] == "List Test Conv"

    def test_get_conversation_detail(
        self,
        backend_url: str,
        auth_headers: dict,
        created_conversation_ids: list[int],
    ) -> None:
        """``GET /api/conversations/{id}`` returns the full detail with
        an empty messages list.

        Returns:
            200 with ``ConversationDetailResponse`` including ``messages``.
        """
        session_id = _make_session_id()
        create_resp = requests.post(
            f"{backend_url}/conversations/",
            headers=auth_headers,
            json={"session_id": session_id, "title": "Detail Test Conv"},
            timeout=10,
        )
        assert create_resp.status_code == 200, create_resp.text
        conv_id = create_resp.json()["id"]
        created_conversation_ids.append(conv_id)

        # Detail
        resp = requests.get(
            f"{backend_url}/conversations/{conv_id}",
            headers=auth_headers,
            timeout=10,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["id"] == conv_id
        assert body["title"] == "Detail Test Conv"
        assert isinstance(body["messages"], list)
        assert len(body["messages"]) == 0

    def test_add_message(
        self,
        backend_url: str,
        auth_headers: dict,
        created_conversation_ids: list[int],
    ) -> None:
        """``POST /api/conversations/message`` adds a message to a
        conversation.

        Returns:
            200 with ``MessageResponse`` containing ``role``, ``content``.
            Subsequent detail shows the message in the list.
        """
        session_id = _make_session_id()
        create_resp = requests.post(
            f"{backend_url}/conversations/",
            headers=auth_headers,
            json={"session_id": session_id, "title": "Message Test Conv"},
            timeout=10,
        )
        assert create_resp.status_code == 200, create_resp.text
        conv_id = create_resp.json()["id"]
        created_conversation_ids.append(conv_id)

        # Add message
        msg_resp = requests.post(
            f"{backend_url}/conversations/message",
            headers=auth_headers,
            json={
                "conversation_id": conv_id,
                "role": "user",
                "content": "Hello from integration test",
            },
            timeout=10,
        )
        assert msg_resp.status_code == 200, msg_resp.text
        msg_body = msg_resp.json()
        assert msg_body["role"] == "user"
        assert msg_body["content"] == "Hello from integration test"

        # Verify via detail
        detail_resp = requests.get(
            f"{backend_url}/conversations/{conv_id}",
            headers=auth_headers,
            timeout=10,
        )
        assert detail_resp.status_code == 200
        messages = detail_resp.json()["messages"]
        assert len(messages) == 1
        assert messages[0]["content"] == "Hello from integration test"

    def test_update_conversation_title(
        self,
        backend_url: str,
        auth_headers: dict,
        created_conversation_ids: list[int],
    ) -> None:
        """``PUT /api/conversations/{id}`` updates the title.

        Returns:
            200 with ``ConversationResponse`` reflecting the new title.
        """
        session_id = _make_session_id()
        create_resp = requests.post(
            f"{backend_url}/conversations/",
            headers=auth_headers,
            json={"session_id": session_id, "title": "Original Title"},
            timeout=10,
        )
        assert create_resp.status_code == 200, create_resp.text
        conv_id = create_resp.json()["id"]
        created_conversation_ids.append(conv_id)

        # Update title
        resp = requests.put(
            f"{backend_url}/conversations/{conv_id}",
            headers=auth_headers,
            json={"title": "Updated Title"},
            timeout=10,
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["title"] == "Updated Title"

    def test_delete_conversation(
        self,
        backend_url: str,
        auth_headers: dict,
    ) -> None:
        """``DELETE /api/conversations/{id}`` removes the conversation.

        Returns:
            200 with success message; subsequent list omits it.
        """
        session_id = _make_session_id()
        create_resp = requests.post(
            f"{backend_url}/conversations/",
            headers=auth_headers,
            json={"session_id": session_id, "title": "Delete Me Conv"},
            timeout=10,
        )
        assert create_resp.status_code == 200, create_resp.text
        conv_id = create_resp.json()["id"]

        # Delete
        del_resp = requests.delete(
            f"{backend_url}/conversations/{conv_id}",
            headers=auth_headers,
            timeout=10,
        )
        assert del_resp.status_code == 200, del_resp.text

        # Verify gone
        list_resp = requests.get(
            f"{backend_url}/conversations/",
            headers=auth_headers,
            timeout=10,
        )
        assert list_resp.status_code == 200
        ids_in_list = [c["id"] for c in list_resp.json()]
        assert conv_id not in ids_in_list, (
            f"Conversation {conv_id} still in list after delete"
        )
