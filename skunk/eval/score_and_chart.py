"""Score golden vs golden-noisy reports for the 10-UID comparison.

Inputs (produced by `eval/run_comparison.py` and the prior 32-UID golden run):
    eval/golden32_final.csv               — golden leg, already scored
    eval/comparison10_noisy_report.csv    — noisy leg, predicted/gold
    eval/comparison10_selection.json      — sampling provenance + per-UID confounders

Outputs:
    eval/score_comparison.csv             — per-UID merged table
    eval/score_comparison.png             — 2-panel chart
    eval/comparison_report.md             — auto-generated dossier
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


_NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def _normalize_scalar(s):
    if s is None: return None
    s = str(s).strip().replace("−", "-").replace(",", "").rstrip("%").strip()
    try: return float(s)
    except ValueError: return None


def _extract_numbers(s):
    if s is None: return []
    return [float(m.replace(",", "")) for m in _NUM_RE.findall(str(s).replace("−", "-"))]


def _close(a, b, rel=1e-3, abs_=1e-3):
    if a == 0 and b == 0: return True
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


def score_row(predicted, gold, failed):
    if failed or predicted is None or str(predicted).strip() == "": return False
    pn, gn = _normalize_scalar(predicted), _normalize_scalar(gold)
    if pn is not None and gn is not None: return _close(pn, gn)
    pl, gl = _extract_numbers(predicted), _extract_numbers(gold)
    if pl and gl and len(pl) == len(gl):
        return all(_close(a, b) for a, b in zip(pl, gl))
    return str(predicted).strip().lower() == str(gold).strip().lower()


def pct_err(predicted, gold, failed):
    if failed: return None
    pl, gl = _extract_numbers(predicted), _extract_numbers(gold)
    if not pl or not gl or len(pl) != len(gl): return None
    return sum(abs(a - b) / abs(b) * 100 if b else abs(a - b)
               for a, b in zip(pl, gl)) / len(pl)


def _tally(df, label):
    pe = pd.to_numeric(df["pct_err"], errors="coerce")
    n = len(df)
    nc = int(df["correct"].sum())
    nf = int(df["failed"].sum())
    w1 = int((df["correct"] | (~df["failed"] & (pe < 1))).sum())
    w5 = int((df["correct"] | (~df["failed"] & (pe < 5))).sum())
    return {"label": label, "n": n, "correct": nc, "within1": w1, "within5": w5,
            "failed": nf, "answered": n - nf}


def main() -> None:
    selection = json.loads(Path("eval/comparison10_selection.json").read_text())
    uids = list(selection["per_uid"].keys())

    golden_all = pd.read_csv("eval/golden32_final.csv")
    golden = golden_all[golden_all["uid"].isin(uids)].copy()
    golden["correct"] = golden["correct"].astype(bool)
    golden["failed"] = golden["failed"].astype(bool)

    noisy = pd.read_csv("eval/comparison10_noisy_report.csv")
    noisy["predicted"] = noisy["predicted"].fillna("")
    noisy["correct"] = noisy.apply(
        lambda r: score_row(r["predicted"], r["gold_answer"], bool(r["failed"])), axis=1)
    noisy["pct_err"] = noisy.apply(
        lambda r: pct_err(r["predicted"], r["gold_answer"], bool(r["failed"])), axis=1)

    # Merge per-UID
    g = golden.set_index("uid")[["predicted", "gold", "correct", "failed", "pct_err", "reason"]]
    g = g.rename(columns={"predicted": "g_pred", "gold": "gold_answer",
                          "correct": "g_correct", "failed": "g_failed",
                          "pct_err": "g_pct_err", "reason": "g_reason"})
    n = noisy.set_index("uid")[["predicted", "gold_answer", "correct", "failed",
                                 "pct_err", "reason", "n_confounders", "noise_seed"]]
    n = n.rename(columns={"predicted": "n_pred", "correct": "n_correct",
                          "failed": "n_failed", "pct_err": "n_pct_err",
                          "reason": "n_reason"})
    n = n.drop(columns=["gold_answer"])
    merged = g.join(n, how="inner").reset_index()
    merged.to_csv("eval/score_comparison.csv", index=False)

    # Headlines
    g_tally = _tally(golden.rename(columns={"gold": "gold_answer"})
                      .assign(correct=golden["correct"], failed=golden["failed"]),
                     "golden")
    n_tally = _tally(noisy, "noisy")

    # ---- chart ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    modes = ["golden", "golden-noisy"]
    correct = [g_tally["correct"], n_tally["correct"]]
    n = g_tally["n"]
    wrong = [n - g_tally["correct"] - g_tally["failed"],
             n - n_tally["correct"] - n_tally["failed"]]
    failed = [g_tally["failed"], n_tally["failed"]]
    x = range(len(modes))
    ax1.bar(x, correct, color="#2a9d8f", label="correct")
    ax1.bar(x, wrong, bottom=correct, color="#e76f51", label="wrong")
    ax1.bar(x, failed, bottom=[c + w for c, w in zip(correct, wrong)],
            color="#264653", label="failed")
    ax1.set_xticks(list(x)); ax1.set_xticklabels(modes)
    ax1.set_ylabel(f"count (of {n})"); ax1.set_ylim(0, n)
    ax1.set_title(f"Outcome breakdown — golden vs golden-noisy (n={n})")
    ax1.legend(loc="upper right")
    for i, (c, w, f) in enumerate(zip(correct, wrong, failed)):
        if c: ax1.text(i, c / 2, str(c), ha="center", va="center", color="white", fontweight="bold")
        if w: ax1.text(i, c + w / 2, str(w), ha="center", va="center", color="white", fontweight="bold")
        if f: ax1.text(i, c + w + f / 2, str(f), ha="center", va="center", color="white", fontweight="bold")

    yi = list(range(len(merged)))
    color_for = {1: "#2a9d8f", 0: "#e76f51", -1: "#264653"}
    g_status = [1 if r["g_correct"] else (-1 if r["g_failed"] else 0) for _, r in merged.iterrows()]
    n_status = [1 if r["n_correct"] else (-1 if r["n_failed"] else 0) for _, r in merged.iterrows()]
    for i, (gs, ns) in enumerate(zip(g_status, n_status)):
        ax2.barh(i - 0.18, 1, height=0.36, color=color_for[gs], left=0)
        ax2.barh(i + 0.18, 1, height=0.36, color=color_for[ns], left=0)
    ax2.set_yticks(yi); ax2.set_yticklabels(merged["uid"].tolist())
    ax2.set_xticks([]); ax2.invert_yaxis()
    ax2.set_title("Per-UID: top = golden, bottom = golden-noisy")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#2a9d8f", label="correct"),
               plt.Rectangle((0, 0), 1, 1, color="#e76f51", label="wrong"),
               plt.Rectangle((0, 0), 1, 1, color="#264653", label="failed")]
    ax2.legend(handles=handles, loc="lower right")
    fig.suptitle("OfficeQA: confounder injection (--golden-noisy) vs clean golden retrieval", fontsize=13)
    fig.tight_layout()
    out = Path("eval/score_comparison.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")

    # ---- markdown report ----
    md = []
    md.append("# OfficeQA: `--golden` vs `--golden-noisy` (10-UID comparison)\n")
    md.append(
        f"10 UIDs from the 32-set, evaluated under two retrieval regimes. Same model "
        f"(`gemini-3-flash-preview`), same config (`max_compute_depth=1`, loosened "
        f"`_VISION_SYSTEM`). Noise sampler: `noise_prob={selection['noise_prob']}`, "
        f"per-UID seed selected (base {selection['base_noise_seed']}) so every UID "
        f"receives ≥1 confounder.\n"
    )
    md.append(
        "- **golden** — gold-truth `PageRef`s only.\n"
        "- **golden-noisy** — gold refs + per-page Bernoulli confounders, drawn 30/70 "
        "from same-bulletin drift / cross-year keyword pool.\n\n"
        "Reports: `eval/golden32_final.csv` (golden), `eval/comparison10_noisy_report.csv` (noisy).\n"
        "Per-UID: `eval/score_comparison.csv`. Chart: `eval/score_comparison.png`. "
        "Sampling provenance: `eval/comparison10_selection.json`.\n"
    )
    md.append("## Headline\n")
    md.append("| Metric                                  | golden          | golden-noisy    |")
    md.append("| --------------------------------------- | --------------: | --------------: |")
    md.append(f"| **Correct** (≤0.1% rel. tol.)           | **{g_tally['correct']} / {n}** | **{n_tally['correct']} / {n}** |")
    md.append(f"| Within 1% drift                         | {g_tally['within1']} / {n}     | {n_tally['within1']} / {n}     |")
    md.append(f"| Within 5% drift                         | {g_tally['within5']} / {n}     | {n_tally['within5']} / {n}     |")
    md.append(f"| Hard failures (no answer)               | {g_tally['failed']} / {n}     | {n_tally['failed']} / {n}     |")
    n_with_conf = sum(1 for v in selection["per_uid"].values() if v["n_confounders"] > 0)
    total_conf = sum(v["n_confounders"] for v in selection["per_uid"].values())
    md.append(f"| UIDs receiving ≥1 confounder            | —               | **{n_with_conf} / {n}** |")
    md.append(f"| Total confounders injected              | —               | **{total_conf}** |")
    md.append("")

    md.append("\n## Per-UID dossier\n")
    for _, r in merged.iterrows():
        u = r["uid"]
        info = selection["per_uid"][u]
        confs = ", ".join(f"`{c['month']} pg{c['page']}`" for c in info["confounders"]) or "(none)"
        md.append(f"### {u}\n")
        md.append(f"- **Confounders ({info['n_confounders']})**: {confs}")
        md.append(f"- **Seed**: {info['noise_seed']}")
        md.append(f"- **Gold**: `{r['gold_answer']}`")
        g_tag = "✓" if r["g_correct"] else ("FAIL" if r["g_failed"] else f"✗ ({r['g_pct_err']:.2f}% err)" if pd.notna(r["g_pct_err"]) else "✗")
        n_tag = "✓" if r["n_correct"] else ("FAIL" if r["n_failed"] else f"✗ ({r['n_pct_err']:.2f}% err)" if pd.notna(r["n_pct_err"]) else "✗")
        md.append(f"- **Golden**: {g_tag} predicted `{r['g_pred']}`" + (f" — {r['g_reason']}" if r["g_failed"] else ""))
        md.append(f"- **Noisy**:  {n_tag} predicted `{r['n_pred']}`" + (f" — {r['n_reason']}" if r["n_failed"] else ""))
        # Status diff
        if r["g_correct"] == r["n_correct"] and r["g_failed"] == r["n_failed"]:
            md.append("- **Status**: same outcome")
        elif r["n_correct"] and not r["g_correct"]:
            md.append("- **Status**: 🟢 noise FIXED a golden miss (stochastic; not necessarily signal)")
        elif r["g_correct"] and not r["n_correct"]:
            md.append("- **Status**: 🔴 noise BROKE a golden win")
        else:
            md.append("- **Status**: 🟡 both incorrect; mode of failure may have shifted")
        md.append("")

    Path("eval/comparison_report.md").write_text("\n".join(md))
    print(f"Wrote eval/score_comparison.csv, eval/score_comparison.png, eval/comparison_report.md")
    print(f"\nGolden:  {g_tally}")
    print(f"Noisy:   {n_tally}")


if __name__ == "__main__":
    main()
