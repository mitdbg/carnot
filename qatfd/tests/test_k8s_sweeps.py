"""The k8s sweep generators and cell records (offline: no S3, no cluster)."""

from __future__ import annotations

import json

import pytest

from qatfd.k8s import __main__ as cli
from qatfd.k8s.benchmarks import benchmark_data
from qatfd.k8s.cells import Cell
from qatfd.k8s.sweeps import bootstrap_enrich_ub


def _plan_args(*extra: str):
    return cli_args("plan", "--sweep", "bootstrap_enrich_ub", "--no-skip-complete", *extra)


def cli_args(*argv: str):
    """Parse with the sweep's own argument group so the defaults are the ones users get."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sweep")
    p.add_argument("--no-skip-complete", action="store_true")
    bootstrap_enrich_ub.add_arguments(p)
    return p.parse_args([a for a in argv if a != "plan"])


def test_default_sweep_is_45_cells_and_nests():
    cells = bootstrap_enrich_ub.cells(_plan_args())
    assert len(cells) == 3 * 3 * 5
    labels = [c.label for c in cells]
    assert labels[0] == "ub_x1_exp1_baseline_s0" and labels[-1] == "ub_x10_exp7_bs_en_s2"
    assert len(set(labels)) == len(labels)
    # X-major, then seed, then cell: every cell at X=1 precedes any X=5 cell
    assert all("_x1_" in lab for lab in labels[:15]) and all("_x5_" in lab for lab in labels[15:30])
    for c in cells:
        assert c.expected_rows == c.meta["x"] * 2
        assert c.system_dir == "search_agent_ub" and c.benchmark == "officeqa"
        assert c.argv[:2] == ["python", "scripts/bootstrap_enrich_upper_bound.py"]


def test_cell_table_overrides():
    cells = {c.meta["cell"]: c for c in bootstrap_enrich_ub.cells(_plan_args("--xs", "5", "--seeds", "0",
                                                                             "--cells", *sorted(bootstrap_enrich_ub.CELLS)))}
    def ov(c: Cell, key: str) -> str:
        vals = [a.split("=", 1)[1] for a in c.argv if a.split("=", 1)[0] == key]
        assert len(vals) == 1, (key, vals)
        return vals[0]
    exp1, exp2, exp7, exp8 = cells["exp1_baseline"], cells["exp2_ws"], cells["exp7_bs_en"], cells["exp8_bs_en_ws"]
    assert ov(exp1, "systems.retrieve.working_set_collection_off") == "true"
    assert ov(exp1, "systems.retrieve.enrich_working_sets") == "null"
    assert ov(exp2, "systems.retrieve.hide_and_clear_working_sets") == "false"
    assert ov(exp2, "+ub.trajectory_working_sets") == "true"
    assert ov(exp7, "systems.retrieve.enrich_working_sets") == "both"
    assert ov(exp7, "systems.retrieve.enrich_query_batch_size") == "5"
    assert ov(exp8, "+ub.trajectory_working_sets") == "true"
    for c in cells.values():
        assert ov(c, "+ub.wipe_collections") == "false"            # pods start on a pristine store
        assert ov(c, "benchmarks.pdf_dir") == "null"                # no PDFs pulled by default
        assert ov(c, "+ub.keep_collections") == "[officeqa-qwen-8b,officeqa-qwen-8b-v1]"
        assert "results_root" not in " ".join(c.argv)               # the chart appends it


def test_qids_mode_and_prefix():
    cells = bootstrap_enrich_ub.cells(_plan_args("--qids", "UID0001", "UID0002", "--xs", "1", "--seeds", "0",
                                                 "--cells", "exp1_baseline", "--run-prefix", "smoke", "--passes", "3"))
    assert [c.label for c in cells] == ["smoke_x1_exp1_baseline_s0"]
    assert cells[0].expected_rows == 2 * 3
    assert "experiments.qids=[UID0001,UID0002]" in cells[0].argv


def test_with_pdfs_pulls_them_and_keeps_pdf_dir():
    c = bootstrap_enrich_ub.cells(_plan_args("--xs", "1", "--seeds", "0", "--cells", "exp1_baseline", "--with-pdfs"))[0]
    assert "officeqa/treasury_bulletin_pdfs/" in c.data
    assert not any(a.startswith("benchmarks.pdf_dir=") for a in c.argv)


def test_values_are_chart_shaped_and_json_serializable():
    c = bootstrap_enrich_ub.cells(_plan_args("--xs", "1", "--seeds", "0", "--cells", "exp1_baseline"))[0]
    v = c.to_values()
    assert set(v) == {"label", "benchmark", "system_dir", "collection", "store_prefix", "data", "argv", "expected_rows"}
    json.dumps(v)
    assert v["collection"] == "officeqa-qwen-8b-v1" and v["store_prefix"] == "officeqa/chromadb"


def test_benchmark_recipes():
    assert benchmark_data("browsecomp_plus").data[-1] == {"src": "browsecomp-plus/browsecomp-plus-element-embeddings/", "include": "metadata_rank*.json"}
    fs = benchmark_data("freshstack", topic="langchain")
    assert fs.collection == "freshstack-langchain-qwen-0.6b-v1" and fs.store_prefix == "freshstack/langchain/chromadb"
    with pytest.raises(ValueError):
        benchmark_data("qampari")


def test_cli_plan_offline(capsys):
    rc = cli.main(["plan", "--sweep", "bootstrap_enrich_ub", "--no-skip-complete", "--xs", "1", "--seeds", "0", "--cells", "exp1_baseline"])
    out = capsys.readouterr().out
    assert rc == 0 and "1/1 cell(s) to run" in out and "ub_x1_exp1_baseline_s0" in out


def test_model_and_reasoning_flags():
    cells = bootstrap_enrich_ub.cells(_plan_args(
        *("--xs 33 --seeds 0 --cells exp5_en --enrich-batch 5 --enrich-max-previous-queries 50 "
          "--model qwen/qwen3.6-35b-a3b --compute-model qwen/qwen3.6-35b-a3b --agent-model qwen/qwen3.6-35b-a3b "
          "--map-model qwen/qwen3.6-35b-a3b --providers parasail,akashml,reka,venice --disable-reasoning --run-prefix dq").split()))
    assert len(cells) == 1
    c = cells[0]
    assert c.label == "dq_x33_exp5_en_s0" and c.expected_rows == 66
    ov = set(c.argv)
    assert "inference.llm_provider_order=[parasail,akashml,reka,venice]" in ov
    assert "systems.retrieve.enrich_config.max_previous_queries=50" in ov
    assert "systems.retrieve.enrich_query_batch_size=5" in ov
    for agent in ("systems.retrieve", "systems.compute", "systems.retrieve.bootstrap_config", "systems.retrieve.enrich_config"):
        assert f"{agent}.disable_reasoning=true" in ov
    assert sum(a.endswith("qwen/qwen3.6-35b-a3b") for a in c.argv) == 6  # search, compute, bootstrap, enrich, 2x map
