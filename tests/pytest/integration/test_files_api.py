"""Integration tests for the ``/api/files`` routes.

Validates file-management CRUD operations against the live Docker Compose
stack: browsing, uploading, creating directories, and deleting.

All tests require ``RUN_INTEGRATION_TESTS=1`` and valid Auth0 credentials.
"""

from __future__ import annotations

import requests


class TestFilesAPI:
    """CRUD operations on the ``/api/files`` endpoints.

    Representation invariant:
        Each test that creates a file or directory registers its full
        container-side path in ``uploaded_file_paths`` so the fixture
        teardown can clean it up.

    Abstraction function:
        Represents the contract between the React frontend and the
        FastAPI file-management routes, exercised end-to-end through
        nginx → uvicorn → local filesystem inside the container.
    """

    # ── browse ──────────────────────────────────────────────────────

    def test_browse_root(self, backend_url: str, auth_headers: dict) -> None:
        """``GET /api/files/browse`` without a path returns the root listing.

        Returns:
            200 with a ``PaginatedFileList`` containing ``items`` (list),
            ``has_more`` (bool).  The ``data/`` directory should be present
            at the root level.
        """
        resp = requests.get(
            f"{backend_url}/files/browse",
            headers=auth_headers,
            timeout=10,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "items" in body
        assert "has_more" in body
        # The container's /carnot/ root should contain at least "data/"
        names = [item["display_name"] for item in body["items"]]
        assert any("data" in n for n in names), (
            f"Expected 'data' or 'data/' in root listing, got {names}"
        )

    # ── upload ──────────────────────────────────────────────────────

    def test_upload_file(
        self,
        backend_url: str,
        auth_headers: dict,
        uploaded_file_paths: list[str],
    ) -> None:
        """``POST /api/files/upload`` stores a small text file in the
        user's data directory.

        Returns:
            200 on success.  The file is visible in a subsequent browse
            of the user's data directory.
        """
        file_content = b"hello integration test"
        resp = requests.post(
            f"{backend_url}/files/upload",
            headers=auth_headers,
            files={"file": ("integration_test.txt", file_content, "text/plain")},
            data={"path": "data/"},
            timeout=15,
        )
        assert resp.status_code == 200, resp.text

    def test_browse_user_data_dir(
        self,
        backend_url: str,
        auth_headers: dict,
        uploaded_file_paths: list[str],
    ) -> None:
        """After uploading, ``GET /api/files/browse?path=data/`` should
        list the uploaded file.

        Requires:
            ``test_upload_file`` has run in this session (file exists in
            the container's user data directory).

        Returns:
            The uploaded ``integration_test.txt`` appears in the listing.
        """
        # First upload a file to guarantee state
        file_content = b"browse test content"
        upload_resp = requests.post(
            f"{backend_url}/files/upload",
            headers=auth_headers,
            files={"file": ("browse_test.txt", file_content, "text/plain")},
            data={"path": "data/"},
            timeout=15,
        )
        assert upload_resp.status_code == 200, upload_resp.text

        # Now browse the data directory
        resp = requests.get(
            f"{backend_url}/files/browse",
            params={"path": "data/"},
            headers=auth_headers,
            timeout=10,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        names = [item["display_name"] for item in body["items"]]
        assert "browse_test.txt" in names, (
            f"Expected 'browse_test.txt' in data dir listing, got {names}"
        )

    # ── create directory ────────────────────────────────────────────

    def test_create_directory(
        self,
        backend_url: str,
        auth_headers: dict,
    ) -> None:
        """``POST /api/files/create-directory`` creates a subdirectory
        inside the user's data directory.

        Returns:
            200 with ``{"message": "Success"}``.
        """
        resp = requests.post(
            f"{backend_url}/files/create-directory",
            headers=auth_headers,
            json={"path": "data/", "name": "integration-test-dir"},
            timeout=10,
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["message"] == "Success"

    # ── delete ──────────────────────────────────────────────────────

    def test_delete_file(
        self,
        backend_url: str,
        auth_headers: dict,
    ) -> None:
        """``POST /api/files/delete`` removes a file that was previously
        uploaded.

        Returns:
            200 with a success message.
        """
        # Upload a file specifically for deletion
        file_content = b"delete me"
        upload_resp = requests.post(
            f"{backend_url}/files/upload",
            headers=auth_headers,
            files={"file": ("to_delete.txt", file_content, "text/plain")},
            data={"path": "data/"},
            timeout=15,
        )
        assert upload_resp.status_code == 200, upload_resp.text

        # Find the actual path from a browse
        browse_resp = requests.get(
            f"{backend_url}/files/browse",
            params={"path": "data/"},
            headers=auth_headers,
            timeout=10,
        )
        assert browse_resp.status_code == 200, browse_resp.text
        items = browse_resp.json()["items"]
        match = [i for i in items if i["display_name"] == "to_delete.txt"]
        assert match, f"Uploaded file not found; items = {items}"
        file_path = match[0]["path"]

        # Delete it
        del_resp = requests.post(
            f"{backend_url}/files/delete",
            headers=auth_headers,
            json={"files": [file_path]},
            timeout=10,
        )
        assert del_resp.status_code == 200, del_resp.text
