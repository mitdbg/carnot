"""Integration tests for the ``/api/datasets`` routes.

Validates dataset CRUD operations against the live Docker Compose stack:
create, list, detail, update, duplicate-name rejection, and delete.

All tests require ``RUN_INTEGRATION_TESTS=1`` and valid Auth0 credentials.
"""

from __future__ import annotations

import requests


def _upload_test_file(backend_url: str, auth_headers: dict, filename: str = "ds_test.txt") -> None:
    """Upload a small helper file so that dataset creation has valid file paths."""
    requests.post(
        f"{backend_url}/files/upload",
        headers=auth_headers,
        files={"file": (filename, b"dataset test content", "text/plain")},
        data={"path": "data/"},
        timeout=15,
    )


def _find_uploaded_path(backend_url: str, auth_headers: dict, filename: str) -> str:
    """Browse the user's data dir and return the container-side path of *filename*."""
    resp = requests.get(
        f"{backend_url}/files/browse",
        params={"path": "data/"},
        headers=auth_headers,
        timeout=10,
    )
    assert resp.status_code == 200, resp.text
    for item in resp.json()["items"]:
        if item["display_name"] == filename:
            return item["path"]
    raise AssertionError(f"{filename!r} not found in data dir listing")


class TestDatasetsAPI:
    """CRUD operations on the ``/api/datasets`` endpoints.

    Representation invariant:
        Every dataset created during a test is registered in
        ``created_dataset_ids`` so the fixture teardown can delete it.

    Abstraction function:
        Represents the contract between the frontend dataset manager and
        the FastAPI dataset routes, exercised through
        nginx → uvicorn → PostgreSQL.
    """

    def test_create_dataset(
        self,
        backend_url: str,
        auth_headers: dict,
        created_dataset_ids: list[int],
    ) -> None:
        """``POST /api/datasets/`` creates a dataset from uploaded files.

        Returns:
            200 with ``DatasetDetailResponse`` containing ``id``, ``name``,
            ``annotation``, ``files`` list.
        """
        _upload_test_file(backend_url, auth_headers, "ds_create.txt")
        file_path = _find_uploaded_path(backend_url, auth_headers, "ds_create.txt")

        resp = requests.post(
            f"{backend_url}/datasets/",
            headers=auth_headers,
            json={
                "name": "integration-test-dataset",
                "shared": False,
                "annotation": "Created by integration test",
                "files": [file_path],
            },
            timeout=15,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["name"] == "integration-test-dataset"
        assert body["annotation"] == "Created by integration test"
        assert "id" in body
        assert len(body["files"]) >= 1
        created_dataset_ids.append(body["id"])

    def test_list_datasets(
        self,
        backend_url: str,
        auth_headers: dict,
        created_dataset_ids: list[int],
    ) -> None:
        """``GET /api/datasets/`` includes a dataset we just created.

        Returns:
            200 with a list of ``DatasetResponse`` objects; the created
            dataset appears with matching ``name`` and ``file_count >= 1``.
        """
        # Ensure a dataset exists
        _upload_test_file(backend_url, auth_headers, "ds_list.txt")
        file_path = _find_uploaded_path(backend_url, auth_headers, "ds_list.txt")
        create_resp = requests.post(
            f"{backend_url}/datasets/",
            headers=auth_headers,
            json={
                "name": "integration-test-list",
                "shared": False,
                "annotation": "list test",
                "files": [file_path],
            },
            timeout=15,
        )
        assert create_resp.status_code == 200, create_resp.text
        ds_id = create_resp.json()["id"]
        created_dataset_ids.append(ds_id)

        # List
        resp = requests.get(
            f"{backend_url}/datasets/",
            headers=auth_headers,
            timeout=10,
        )
        assert resp.status_code == 200, resp.text
        datasets = resp.json()
        match = [d for d in datasets if d["id"] == ds_id]
        assert len(match) == 1, f"Dataset {ds_id} not found in list"
        assert match[0]["file_count"] >= 1

    def test_get_dataset_detail(
        self,
        backend_url: str,
        auth_headers: dict,
        created_dataset_ids: list[int],
    ) -> None:
        """``GET /api/datasets/{id}`` returns the full detail with files.

        Returns:
            200 with ``DatasetDetailResponse`` including ``files`` list
            and ``annotation``.
        """
        _upload_test_file(backend_url, auth_headers, "ds_detail.txt")
        file_path = _find_uploaded_path(backend_url, auth_headers, "ds_detail.txt")
        create_resp = requests.post(
            f"{backend_url}/datasets/",
            headers=auth_headers,
            json={
                "name": "integration-test-detail",
                "shared": False,
                "annotation": "detail test",
                "files": [file_path],
            },
            timeout=15,
        )
        assert create_resp.status_code == 200, create_resp.text
        ds_id = create_resp.json()["id"]
        created_dataset_ids.append(ds_id)

        # Detail
        resp = requests.get(
            f"{backend_url}/datasets/{ds_id}",
            headers=auth_headers,
            timeout=10,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["id"] == ds_id
        assert body["annotation"] == "detail test"
        assert isinstance(body["files"], list)
        assert len(body["files"]) >= 1

    def test_update_dataset_annotation(
        self,
        backend_url: str,
        auth_headers: dict,
        created_dataset_ids: list[int],
    ) -> None:
        """``PUT /api/datasets/{id}`` updates the annotation.

        Returns:
            200 with ``DatasetDetailResponse`` reflecting the new annotation.
        """
        _upload_test_file(backend_url, auth_headers, "ds_update.txt")
        file_path = _find_uploaded_path(backend_url, auth_headers, "ds_update.txt")
        create_resp = requests.post(
            f"{backend_url}/datasets/",
            headers=auth_headers,
            json={
                "name": "integration-test-update",
                "shared": False,
                "annotation": "original",
                "files": [file_path],
            },
            timeout=15,
        )
        assert create_resp.status_code == 200, create_resp.text
        ds_id = create_resp.json()["id"]
        created_dataset_ids.append(ds_id)

        # Update annotation
        resp = requests.put(
            f"{backend_url}/datasets/{ds_id}",
            headers=auth_headers,
            json={"annotation": "updated annotation"},
            timeout=10,
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["annotation"] == "updated annotation"

    def test_duplicate_name_rejected(
        self,
        backend_url: str,
        auth_headers: dict,
        created_dataset_ids: list[int],
    ) -> None:
        """A second ``POST /api/datasets/`` with the same name returns 400.

        Returns:
            400 with ``"Dataset name already exists"`` detail.
        """
        _upload_test_file(backend_url, auth_headers, "ds_dup.txt")
        file_path = _find_uploaded_path(backend_url, auth_headers, "ds_dup.txt")
        payload = {
            "name": "integration-test-duplicate",
            "shared": False,
            "annotation": "dup test",
            "files": [file_path],
        }

        # First create
        resp1 = requests.post(
            f"{backend_url}/datasets/",
            headers=auth_headers,
            json=payload,
            timeout=15,
        )
        assert resp1.status_code == 200, resp1.text
        created_dataset_ids.append(resp1.json()["id"])

        # Second create with same name → 400
        resp2 = requests.post(
            f"{backend_url}/datasets/",
            headers=auth_headers,
            json=payload,
            timeout=15,
        )
        assert resp2.status_code == 400, f"Expected 400, got {resp2.status_code}: {resp2.text}"
        assert "already exists" in resp2.json()["detail"].lower()

    def test_delete_dataset(
        self,
        backend_url: str,
        auth_headers: dict,
    ) -> None:
        """``DELETE /api/datasets/{id}`` removes the dataset.

        Returns:
            200 with success message; subsequent list omits the dataset.
        """
        _upload_test_file(backend_url, auth_headers, "ds_delete.txt")
        file_path = _find_uploaded_path(backend_url, auth_headers, "ds_delete.txt")
        create_resp = requests.post(
            f"{backend_url}/datasets/",
            headers=auth_headers,
            json={
                "name": "integration-test-delete",
                "shared": False,
                "annotation": "delete test",
                "files": [file_path],
            },
            timeout=15,
        )
        assert create_resp.status_code == 200, create_resp.text
        ds_id = create_resp.json()["id"]

        # Delete
        del_resp = requests.delete(
            f"{backend_url}/datasets/{ds_id}",
            headers=auth_headers,
            timeout=10,
        )
        assert del_resp.status_code == 200, del_resp.text

        # Verify it's gone
        list_resp = requests.get(
            f"{backend_url}/datasets/",
            headers=auth_headers,
            timeout=10,
        )
        assert list_resp.status_code == 200
        ids_in_list = [d["id"] for d in list_resp.json()]
        assert ds_id not in ids_in_list, f"Dataset {ds_id} still in list after delete"
