"""Unit tests for qatfd.keys — the shared corpus-key helpers (see CORPUS_MODEL.md).

Run from the qatfd dir: python3 -m pytest tests/
"""

from qatfd.keys import (
    FINANCEBENCH_PAGE_SEP,
    financebench_doc,
    financebench_page_key,
    freshstack_byte_range,
    freshstack_file_id,
    officeqa_doc,
)


def test_financebench_page_key_roundtrip():
    key = financebench_page_key("3M_2018_10K", 59)
    assert key == "3M_2018_10K::p59"
    assert financebench_doc(key) == "3M_2018_10K"


def test_financebench_doc_names_with_underscores_and_digits():
    # doc_names contain underscores and digits; only the ::p separator splits.
    key = financebench_page_key("AMERICANWATERWORKS_2021_10K", 0)
    assert financebench_doc(key) == "AMERICANWATERWORKS_2021_10K"
    assert FINANCEBENCH_PAGE_SEP not in financebench_doc(key)


def test_officeqa_doc_collapses_page_to_bulletin():
    assert officeqa_doc("1946_11_41") == "1946_11"
    assert officeqa_doc("2002_12_8") == "2002_12"


def test_freshstack_file_id_strips_byte_range():
    assert freshstack_file_id("azure-openai/LICENSE.md_0_1140") == "azure-openai/LICENSE.md"
    # paths with underscores: only a trailing _{int}_{int} is stripped.
    assert freshstack_file_id("repo/my_file_name.py_100_200") == "repo/my_file_name.py"
    # no byte-range suffix -> unchanged (already a file id).
    assert freshstack_file_id("azure-openai/LICENSE.md") == "azure-openai/LICENSE.md"


def test_freshstack_byte_range():
    assert freshstack_byte_range("azure-openai/LICENSE.md_5088_13656") == (5088, 13656)
    assert freshstack_byte_range("azure-openai/LICENSE.md") is None


def test_freshstack_file_id_idempotent_on_file_ids():
    # recall_metrics collapses the retrieved side unconditionally; a file id must map to itself.
    fid = freshstack_file_id("framework/phpstan.src.neon.dist_0_569")
    assert freshstack_file_id(fid) == fid
