"""Analyze Wikidata KG matching quality across the postprocess pipeline.

Reads the step-wise intermediate JSON files and the ontology coverage report
to produce a comprehensive quality report with statistics on:
  - Canonicalization match rate (resolved vs unresolved)
  - Materialization expansion statistics
  - Label resolution quality (QIDs that stayed as raw text)
  - Type-constraint quality (semantic relevance of profiler choices)
  - Per-concept-key breakdown
  - Concrete failure examples

Output: kg_quality_report.json (machine-readable) + kg_quality_report.txt (human-readable)
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

HERE = Path(__file__).resolve().parent
_QID_RE = re.compile(r"^Q[1-9]\d*$")


def _load(name: str) -> Dict[str, Any]:
    p = HERE / name
    if not p.exists():
        print(f"WARNING: {name} not found", file=sys.stderr)
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _is_qid(v: str) -> bool:
    return bool(_QID_RE.match(str(v).strip()))


def _flatten_values(results: Dict[str, Dict[str, Any]], keys: Optional[Set[str]] = None) -> Dict[str, List[str]]:
    """Return {concept_key: [all_values_across_docs]}."""
    out: Dict[str, List[str]] = defaultdict(list)
    for doc_vals in results.values():
        for k, v in doc_vals.items():
            if keys and k not in keys:
                continue
            if isinstance(v, list):
                for x in v:
                    if x is not None:
                        out[k].append(str(x))
            elif v is not None:
                out[k].append(str(v))
    return dict(out)


def _unique_values(values: List[str]) -> Tuple[int, int]:
    """Return (total, unique) counts."""
    return len(values), len(set(values))


def analyze_canonicalization(
    normalized: Dict[str, Dict[str, Any]],
    canonicalized: Dict[str, Dict[str, Any]],
    str_keys: Set[str],
) -> Dict[str, Any]:
    """Compare normalized -> canonicalized to measure QID resolution rates."""
    norm_flat = _flatten_values(normalized, str_keys)
    canon_flat = _flatten_values(canonicalized, str_keys)

    per_key: Dict[str, Dict[str, Any]] = {}
    total_resolved = 0
    total_unresolved = 0
    total_values = 0

    unresolved_examples: Dict[str, List[str]] = defaultdict(list)

    for key in sorted(str_keys):
        norm_vals = norm_flat.get(key, [])
        canon_vals = canon_flat.get(key, [])
        unique_norm = set(norm_vals)
        unique_canon = set(canon_vals)

        qids = {v for v in unique_canon if _is_qid(v)}
        non_qids = unique_canon - qids
        resolved = len(qids)
        unresolved = len(non_qids)

        per_key[key] = {
            "total_value_occurrences": len(canon_vals),
            "unique_input_values": len(unique_norm),
            "unique_output_values": len(unique_canon),
            "resolved_to_qid": resolved,
            "unresolved_kept_text": unresolved,
            "resolution_rate": round(resolved / max(1, resolved + unresolved), 4),
        }

        total_resolved += resolved
        total_unresolved += unresolved
        total_values += len(canon_vals)

        for v in sorted(non_qids)[:10]:
            unresolved_examples[key].append(v)

    return {
        "summary": {
            "total_unique_values": total_resolved + total_unresolved,
            "resolved_to_qid": total_resolved,
            "unresolved_kept_text": total_unresolved,
            "overall_resolution_rate": round(
                total_resolved / max(1, total_resolved + total_unresolved), 4
            ),
            "total_value_occurrences": total_values,
        },
        "per_key": per_key,
        "unresolved_examples": dict(unresolved_examples),
    }


def analyze_materialization(
    canonicalized: Dict[str, Dict[str, Any]],
    materialized: Dict[str, Dict[str, Any]],
    str_keys: Set[str],
) -> Dict[str, Any]:
    """Measure how much materialization expanded QID sets."""
    canon_flat = _flatten_values(canonicalized, str_keys)
    mat_flat = _flatten_values(materialized, str_keys)

    per_key: Dict[str, Dict[str, Any]] = {}
    total_before = 0
    total_after = 0

    for key in sorted(str_keys):
        before = set(canon_flat.get(key, []))
        after = set(mat_flat.get(key, []))
        added = after - before
        removed = before - after

        qids_before = {v for v in before if _is_qid(v)}
        qids_after = {v for v in after if _is_qid(v)}
        qids_added = qids_after - qids_before

        per_key[key] = {
            "unique_before": len(before),
            "unique_after": len(after),
            "values_added": len(added),
            "values_removed": len(removed),
            "qids_before": len(qids_before),
            "qids_after": len(qids_after),
            "ancestor_qids_added": len(qids_added),
            "expansion_ratio": round(len(after) / max(1, len(before)), 3),
        }
        total_before += len(before)
        total_after += len(after)

    return {
        "summary": {
            "total_unique_before": total_before,
            "total_unique_after": total_after,
            "net_added": total_after - total_before,
            "expansion_ratio": round(total_after / max(1, total_before), 3),
        },
        "per_key": per_key,
    }


def analyze_label_resolution(
    materialized: Dict[str, Dict[str, Any]],
    resolved: Dict[str, Dict[str, Any]],
    str_keys: Set[str],
) -> Dict[str, Any]:
    """Check label resolution: QIDs that failed to get labels."""
    mat_flat = _flatten_values(materialized, str_keys)
    res_flat = _flatten_values(resolved, str_keys)

    per_key: Dict[str, Dict[str, Any]] = {}
    total_qid_in = 0
    total_labeled = 0
    total_passthrough_qid = 0

    for key in sorted(str_keys):
        mat_vals = set(mat_flat.get(key, []))
        res_vals = set(res_flat.get(key, []))

        qids_in = {v for v in mat_vals if _is_qid(v)}
        qids_still_present = {v for v in res_vals if _is_qid(v)}
        labeled = len(qids_in) - len(qids_still_present)

        per_key[key] = {
            "qids_input": len(qids_in),
            "qids_labeled": labeled,
            "qids_still_as_qid": len(qids_still_present),
            "label_rate": round(labeled / max(1, len(qids_in)), 4),
            "non_qid_values": len(mat_vals - qids_in),
        }
        total_qid_in += len(qids_in)
        total_labeled += labeled
        total_passthrough_qid += len(qids_still_present)

    return {
        "summary": {
            "total_qids_input": total_qid_in,
            "total_labeled": total_labeled,
            "total_passthrough_qid": total_passthrough_qid,
            "label_rate": round(total_labeled / max(1, total_qid_in), 4),
        },
        "per_key": per_key,
    }


def analyze_type_constraints(coverage_report: Dict[str, Any]) -> Dict[str, Any]:
    """Audit the concept profiler's type constraint choices for semantic relevance."""
    entries = coverage_report.get("entries", [])
    profiles = coverage_report.get("profiling_report", {}).get("profiles", {})

    issues: List[Dict[str, Any]] = []
    stats = {
        "total_concept_keys": len(entries),
        "enabled_keys": 0,
        "disabled_keys": 0,
        "avg_confidence": 0.0,
        "low_confidence_keys": [],
        "wrong_type_constraint_suspects": [],
    }

    confidences = []

    known_problematic = {
        "animal:taxonomic-group": "type constraints include 'musical group' (Q215380) and 'geological group' (Q824979) — neither related to biological taxonomy",
        "book:nationality": "type constraints include 'individual copy of a book' (Q53731850) and 'nationality for sports' (Q3337001) — not about nationalities/countries",
        "plant:attribute": "type constraints include 'heraldic attribute' (Q834104) and 'factory' (Q83405) — irrelevant to plant biology",
        "plant:location": "type constraints include 'place of death' (Q18658526) and 'location of burial' (Q12131650) — not about geographic regions",
        "novel:attribute": "type constraints include 'heraldic attribute' (Q834104) — irrelevant to novel properties",
        "book:subject": "type constraints include grammatical 'subject' (Q164573) and 'school subject' (Q362165) — not about book topics/themes",
        "film:setting": "type constraints include 'thin film' (Q1137203) and 'film director' (Q2526255) — not about geographic locations",
        "film:nationality": "type constraints include 'thin film' (Q1137203) — unrelated to country of origin",
        "book:time-period": "type constraints include 'Time Period in Chrono Trigger' (Q12737784) — a video game concept, not a historical period",
        "film:time-period": "type constraints include 'Time Period in Chrono Trigger' (Q12737784) — a video game concept",
    }

    for entry in entries:
        key = entry.get("concept_key", "")
        reason = entry.get("reason")
        conf = entry.get("confidence", 0)

        if reason in ("no_profile", "resolve_disabled"):
            stats["disabled_keys"] += 1
            if conf > 0:
                confidences.append(conf)
            continue

        stats["enabled_keys"] += 1
        confidences.append(conf)

        if conf < 0.6:
            stats["low_confidence_keys"].append({"key": key, "confidence": conf})

        if key in known_problematic:
            scope_labels = {}
            prof = profiles.get(key, {})
            evidence = prof.get("evidence", {})
            scope_labels = evidence.get("chosen_scope_labels", {})
            stats["wrong_type_constraint_suspects"].append({
                "key": key,
                "confidence": conf,
                "type_constraints": entry.get("type_constraints", []),
                "scope_labels": scope_labels,
                "issue": known_problematic[key],
            })

    stats["avg_confidence"] = round(sum(confidences) / max(1, len(confidences)), 4) if confidences else 0

    return stats


def analyze_resolution_errors(
    normalized: Dict[str, Dict[str, Any]],
    canonicalized: Dict[str, Dict[str, Any]],
    resolved_labels: Dict[str, Dict[str, Any]],
    str_keys: Set[str],
) -> Dict[str, Any]:
    """Find concrete examples where Wikidata resolution produced wrong results.
    
    Compares input value (after normalization) with the final label to detect
    semantic drift (e.g., "sea" -> "Southeast Asia").
    """
    mismatches: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for doc_id in normalized:
        norm_doc = normalized.get(doc_id, {})
        canon_doc = canonicalized.get(doc_id, {})
        label_doc = resolved_labels.get(doc_id, {})

        for key in str_keys:
            norm_val = norm_doc.get(key)
            canon_val = canon_doc.get(key)
            label_val = label_doc.get(key)

            if norm_val is None or canon_val is None or label_val is None:
                continue

            norm_list = norm_val if isinstance(norm_val, list) else [norm_val]
            canon_list = canon_val if isinstance(canon_val, list) else [canon_val]
            label_list = label_val if isinstance(label_val, list) else [label_val]

            for nv, cv, lv in zip(norm_list, canon_list, label_list):
                if nv is None or cv is None or lv is None:
                    continue
                nv_s = str(nv).strip().lower()
                lv_s = str(lv).strip().lower()
                cv_s = str(cv).strip()

                if not _is_qid(cv_s):
                    continue
                if nv_s == lv_s:
                    continue

                if nv_s in lv_s or lv_s in nv_s:
                    continue

                if len(mismatches[key]) < 15:
                    mismatches[key].append({
                        "input": nv_s,
                        "qid": cv_s,
                        "resolved_label": lv_s,
                    })

    deduped: Dict[str, List[Dict[str, str]]] = {}
    for key, examples in mismatches.items():
        seen: Set[str] = set()
        unique = []
        for ex in examples:
            sig = f"{ex['input']}|{ex['qid']}"
            if sig not in seen:
                seen.add(sig)
                unique.append(ex)
        deduped[key] = unique

    total_suspect = sum(len(v) for v in deduped.values())
    return {
        "total_suspect_mismatches": total_suspect,
        "per_key": {k: {"count": len(v), "examples": v} for k, v in sorted(deduped.items())},
    }


def analyze_tag_coverage(step7: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the final expanded tag statistics."""
    stats_raw = step7.get("stats", {})
    results = step7.get("results", {})
    schema = step7.get("schema", [])

    total_tags = len(stats_raw)
    total_docs = len(results) if results else 0

    present_counts = []
    for tag, st in stats_raw.items():
        present_counts.append(int(st.get("present", 0)))

    freq_histogram = Counter()
    for p in present_counts:
        if p == 0:
            freq_histogram["0_never_used"] += 1
        elif p == 1:
            freq_histogram["1_singleton"] += 1
        elif p <= 3:
            freq_histogram["2-3_rare"] += 1
        elif p <= 10:
            freq_histogram["4-10_uncommon"] += 1
        elif p <= 50:
            freq_histogram["11-50_moderate"] += 1
        else:
            freq_histogram["50+_frequent"] += 1

    return {
        "total_tags": total_tags,
        "total_docs": total_docs,
        "frequency_histogram": dict(sorted(freq_histogram.items())),
        "tags_with_0_docs": freq_histogram.get("0_never_used", 0),
        "singleton_tags": freq_histogram.get("1_singleton", 0),
        "tags_above_min_frequency_3": sum(1 for p in present_counts if p >= 3),
    }


def generate_report():
    raw_data = _load("quest_sem_map_output_clean.json")
    step2 = _load("step2_normalized.json")
    step3 = _load("step3_canonicalized.json")
    step5 = _load("step5_materialized.json")
    step6 = _load("step6_resolved_to_labels.json")
    step7 = _load("step7_expanded.json")
    coverage = _load("ontology_coverage_report.json")
    metrics = _load("postprocess_metrics.json")

    schema_cols = raw_data.get("concept_schema_cols", [])
    str_keys: Set[str] = set()
    for col in schema_cols:
        tp = col.get("type", "")
        if tp in ("str", "List[str]"):
            str_keys.add(col["name"])

    norm_results = step2.get("results", {})
    canon_results = step3.get("results", {})
    mat_results = step5.get("results", {})
    label_results = step6.get("results", {})

    canon_analysis = analyze_canonicalization(norm_results, canon_results, str_keys)
    mat_analysis = analyze_materialization(canon_results, mat_results, str_keys)
    label_analysis = analyze_label_resolution(mat_results, label_results, str_keys)
    type_analysis = analyze_type_constraints(coverage)
    error_analysis = analyze_resolution_errors(norm_results, canon_results, label_results, str_keys)
    tag_analysis = analyze_tag_coverage(step7)

    report = {
        "pipeline_metrics": metrics,
        "canonicalization": canon_analysis,
        "materialization": mat_analysis,
        "label_resolution": label_analysis,
        "type_constraint_quality": type_analysis,
        "resolution_errors": error_analysis,
        "tag_coverage": tag_analysis,
    }

    (HERE / "kg_quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = format_text_report(report)
    (HERE / "kg_quality_report.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"Reports written to {HERE / 'kg_quality_report.json'}")
    print(f"                and {HERE / 'kg_quality_report.txt'}")
    return report


def format_text_report(report: Dict[str, Any]) -> List[str]:
    L: List[str] = []
    W = 80

    def section(title: str):
        L.append("")
        L.append("=" * W)
        L.append(f"  {title}")
        L.append("=" * W)

    def subsection(title: str):
        L.append("")
        L.append(f"--- {title} ---")

    L.append("=" * W)
    L.append("  WIKIDATA KG MATCHING QUALITY REPORT")
    L.append("=" * W)

    pm = report.get("pipeline_metrics", {})
    L.append(f"  Raw values:           {pm.get('raw_value_count', '?'):>8,}")
    L.append(f"  After canonicalize:   {pm.get('canonicalized_value_count', '?'):>8,}")
    L.append(f"  After propagation:    {pm.get('propagated_value_count', '?'):>8,}")
    L.append(f"  After materialization: {pm.get('materialized_value_count', '?'):>8,}")
    L.append(f"  Alias collapse:       {pm.get('alias_collapse_count', '?'):>8,}")
    L.append(f"  Materialization added: {pm.get('materialization_added_count', '?'):>8,}")
    L.append(f"  Propagation added:    {pm.get('cross_concept_propagation_added_count', '?'):>8,}")
    L.append(f"  Active tags:          {pm.get('active_tag_count', '?'):>8,}")

    # ── CANONICALIZATION ─────────────────────────────────────────
    section("1. CANONICALIZATION (Wikidata QID Resolution)")
    cs = report["canonicalization"]["summary"]
    L.append(f"  Unique string values:      {cs['total_unique_values']:>6,}")
    L.append(f"  Resolved to QID:           {cs['resolved_to_qid']:>6,}  ({cs['overall_resolution_rate']:.1%})")
    L.append(f"  Unresolved (kept as text): {cs['unresolved_kept_text']:>6,}  ({1-cs['overall_resolution_rate']:.1%})")
    L.append(f"  Total value occurrences:   {cs['total_value_occurrences']:>6,}")

    subsection("Per concept key")
    for key, stats in sorted(report["canonicalization"]["per_key"].items()):
        rate = stats["resolution_rate"]
        bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
        L.append(f"  {key:<35s} {bar} {rate:5.1%}  ({stats['resolved_to_qid']}/{stats['resolved_to_qid']+stats['unresolved_kept_text']} unique)")

    subsection("Unresolved value examples (kept as plain text)")
    for key, examples in sorted(report["canonicalization"]["unresolved_examples"].items()):
        L.append(f"  {key}:")
        for ex in examples[:8]:
            L.append(f"    - \"{ex}\"")

    # ── MATERIALIZATION ──────────────────────────────────────────
    section("2. MATERIALIZATION (Ancestor Expansion)")
    ms = report["materialization"]["summary"]
    L.append(f"  Unique values before:  {ms['total_unique_before']:>6,}")
    L.append(f"  Unique values after:   {ms['total_unique_after']:>6,}")
    L.append(f"  Net added:             {ms['net_added']:>6,}")
    L.append(f"  Expansion ratio:       {ms['expansion_ratio']:>6.2f}x")

    subsection("Per concept key")
    for key, stats in sorted(report["materialization"]["per_key"].items()):
        if stats["values_added"] > 0 or stats["unique_before"] > 0:
            L.append(
                f"  {key:<35s}  {stats['unique_before']:>4} -> {stats['unique_after']:>4}  "
                f"(+{stats['values_added']}, {stats['expansion_ratio']:.2f}x)"
            )

    # ── LABEL RESOLUTION ─────────────────────────────────────────
    section("3. LABEL RESOLUTION (QID -> Human Label)")
    ls = report["label_resolution"]["summary"]
    L.append(f"  QIDs input:            {ls['total_qids_input']:>6,}")
    L.append(f"  Successfully labeled:  {ls['total_labeled']:>6,}  ({ls['label_rate']:.1%})")
    L.append(f"  Still showing as QID:  {ls['total_passthrough_qid']:>6,}")

    # ── TYPE CONSTRAINT QUALITY ──────────────────────────────────
    section("4. TYPE CONSTRAINT QUALITY (Concept Profiler)")
    tc = report["type_constraint_quality"]
    L.append(f"  Total concept keys:    {tc['total_concept_keys']:>6}")
    L.append(f"  Enabled (resolved):    {tc['enabled_keys']:>6}")
    L.append(f"  Disabled:              {tc['disabled_keys']:>6}")
    L.append(f"  Average confidence:    {tc['avg_confidence']:>6.3f}")

    if tc.get("low_confidence_keys"):
        subsection("Low confidence keys (< 0.6)")
        for item in tc["low_confidence_keys"]:
            L.append(f"  {item['key']:<35s}  conf={item['confidence']:.4f}")

    if tc.get("wrong_type_constraint_suspects"):
        subsection("PROBLEMATIC type constraints (semantic mismatch)")
        for item in tc["wrong_type_constraint_suspects"]:
            L.append(f"  {item['key']} (conf={item['confidence']:.3f}):")
            L.append(f"    Issue: {item['issue']}")
            if item.get("scope_labels"):
                for qid, label in item["scope_labels"].items():
                    L.append(f"    - {qid}: \"{label}\"")

    # ── RESOLUTION ERRORS ────────────────────────────────────────
    section("5. RESOLUTION ERRORS (Input -> Wrong Label)")
    re_data = report["resolution_errors"]
    L.append(f"  Total suspect mismatches: {re_data['total_suspect_mismatches']}")
    L.append("")
    L.append("  These are cases where the input string was resolved to a QID,")
    L.append("  but the final label has NO overlap with the original text,")
    L.append("  suggesting the wrong Wikidata entity was chosen.")

    for key, info in sorted(re_data["per_key"].items()):
        if info["count"] == 0:
            continue
        L.append(f"")
        L.append(f"  {key} ({info['count']} suspect mismatches):")
        for ex in info["examples"][:8]:
            L.append(f'    "{ex["input"]}" -> {ex["qid"]} -> "{ex["resolved_label"]}"')

    # ── TAG COVERAGE ─────────────────────────────────────────────
    section("6. FINAL TAG COVERAGE")
    tg = report["tag_coverage"]
    L.append(f"  Total boolean tags:       {tg['total_tags']:>6,}")
    L.append(f"  Total documents:          {tg['total_docs']:>6,}")
    L.append(f"  Tags never used (0 docs): {tg['tags_with_0_docs']:>6,}")
    L.append(f"  Singleton tags (1 doc):   {tg['singleton_tags']:>6,}")
    L.append(f"  Tags >= 3 docs (usable):  {tg['tags_above_min_frequency_3']:>6,}")

    subsection("Frequency distribution")
    for bucket, count in sorted(tg["frequency_histogram"].items()):
        bar = "█" * min(60, max(1, count // 50))
        L.append(f"  {bucket:<20s}  {count:>6,}  {bar}")

    # ── OVERALL ASSESSMENT ───────────────────────────────────────
    section("7. OVERALL ASSESSMENT")
    res_rate = cs["overall_resolution_rate"]
    n_bad_constraints = len(tc.get("wrong_type_constraint_suspects", []))
    n_mismatches = re_data["total_suspect_mismatches"]

    L.append("")
    L.append(f"  Resolution rate: {res_rate:.1%}")
    if res_rate >= 0.7:
        L.append("  -> Decent raw match rate, but quality depends on correctness.")
    elif res_rate >= 0.4:
        L.append("  -> Moderate. Significant portion of values remain as free text.")
    else:
        L.append("  -> LOW. Most values failed to resolve to Wikidata entities.")

    L.append(f"")
    L.append(f"  Type constraint problems: {n_bad_constraints} / {tc['enabled_keys']} concept keys")
    L.append(f"  Resolution mismatches:    {n_mismatches} (input != output label)")
    L.append("")

    if n_bad_constraints >= tc["enabled_keys"] * 0.5:
        L.append("  ** CRITICAL: Majority of concept keys have wrong type constraints. **")
        L.append("  The concept profiler is resolving concept-key names against Wikidata")
        L.append("  and finding entities that match the surface words (e.g., 'group',")
        L.append("  'attribute') rather than the intended domain-specific semantics.")
        L.append("  This causes the type-validation filter to reject correct entities")
        L.append("  and accept wrong ones during canonicalization.")
    L.append("")

    if n_mismatches > 30:
        L.append("  ** WARNING: Many resolution mismatches detected. **")
        L.append("  Values like 'sea' being resolved to 'Southeast Asia' or")
        L.append("  'college graduate' to a Chinese journal title indicate that")
        L.append("  wrong QIDs are being selected due to type-constraint errors.")
    L.append("")

    usable_pct = tg["tags_above_min_frequency_3"] / max(1, tg["total_tags"])
    L.append(f"  Usable tags (freq >= 3): {tg['tags_above_min_frequency_3']:,} / {tg['total_tags']:,} ({usable_pct:.1%})")
    if usable_pct < 0.2:
        L.append("  -> Very sparse: most tags appear in < 3 documents.")
        L.append("  This limits the query planner's ability to build useful filters.")
    L.append("")

    section("8. RECOMMENDATIONS")
    L.append("")
    L.append("  1. FIX TYPE CONSTRAINTS: The concept profiler uses surface-word")
    L.append("     matching on concept-key names (e.g., 'plant:attribute') to find")
    L.append("     Wikidata scope entities. This produces irrelevant types like")
    L.append("     'heraldic attribute' or 'factory' for plant biology. Consider:")
    L.append("     - Using domain-aware prompting to find correct Wikidata types")
    L.append("     - Adding manual overrides via ontology_config for known domains")
    L.append("     - Incorporating actual sample values into type constraint inference")
    L.append("")
    L.append("  2. VALIDATE RESOLUTIONS: Add a post-canonicalization check that")
    L.append("     compares input string similarity with resolved QID labels.")
    L.append("     Flag cases where edit distance / overlap is very low.")
    L.append("")
    L.append("  3. IMPROVE PROPAGATION: Cross-concept propagation added 0 values.")
    L.append("     The inferred_cross_concept_hierarchy is empty. Consider manual")
    L.append("     hierarchy edges (e.g., bird -> animal, novel -> book).")
    L.append("")
    L.append("  4. REDUCE TAG SPARSITY: With only ~{:.0f}% usable tags, the filter".format(usable_pct * 100))
    L.append("     catalog is very sparse. Aggressive materialization or broader")
    L.append("     type constraints could help, but only if resolution quality")
    L.append("     improves first.")
    L.append("")

    return L


if __name__ == "__main__":
    generate_report()
