"""
Evidence analysis:
1. For hard queries: do gold docs' TEXT contain the QUEST category terms?
   Compare with: does our METADATA contain them?
2. For easy queries: same check.
3. Domain-level coverage statistics.
4. Characterize where metadata helps vs hurts.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

EVAL_DIR = Path(__file__).parent / "eval_results"
TMP_DIR = Path(__file__).parent / "tmp"
META_DIR = Path(__file__).parent / "results_text-embedding-3-large"
SEM_MAP_DIR = Path(__file__).parent
SUBSETS = [1, 2, 3]

# ── Loaders ──────────────────────────────────────────────────────────────────

def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return [r for r in records if not r.get("_summary")]


def load_queries(subset):
    with open(TMP_DIR / f"subset_{subset}_quest_queries.jsonl") as f:
        return [json.loads(l) for l in f]


def load_documents(subset):
    """Load corpus documents: title -> text."""
    path = TMP_DIR / f"subset_{subset}_documents.jsonl"
    docs = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            title = d.get("title") or d.get("metadata", {}).get("title", "")
            text = d.get("text", "")
            if title:
                if title not in docs:
                    docs[title] = text
                else:
                    docs[title] += " " + text  # concatenate chunks
    return docs


def load_metadata(subset):
    with open(SEM_MAP_DIR / f"sem_map_subset_{subset}/postprocess_step3_augmented.json") as f:
        return json.load(f)


def find_entity(metadata, title):
    """Find entity in metadata by title."""
    for eid, facets in metadata.items():
        if not isinstance(facets, dict):
            continue
        eid_base = eid.rsplit("-", 1)[0].replace("_", " ")
        if eid_base.lower() == title.lower():
            return eid, facets
    # Fuzzy fallback
    slug = title.lower().replace(" ", "_")
    for eid, facets in metadata.items():
        if not isinstance(facets, dict):
            continue
        if slug[:18] in eid.lower():
            return eid, facets
    return None, None


def extract_mark_terms(original_query):
    """Extract terms from <mark>...</mark> in original query."""
    return re.findall(r"<mark>(.*?)</mark>", original_query)


def get_all_metadata_values(facets):
    """Get all string values from all facets."""
    vals = set()
    for k, v in facets.items():
        if isinstance(v, list):
            vals.update(str(x).lower() for x in v)
        elif isinstance(v, str):
            vals.add(v.lower())
        elif isinstance(v, (int, float)):
            vals.add(str(v).lower())
    return vals


def term_in_text(term, text):
    """Check if term appears in text (case-insensitive)."""
    return term.lower() in text.lower()


def term_in_metadata(term, meta_vals):
    """Check if term appears in any metadata value."""
    t = term.lower()
    return any(t in val for val in meta_vals)


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: Concrete examples — hard queries
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 90)
print("PART 1: Hard query examples — are QUEST category terms in document TEXT or METADATA?")
print("=" * 90)

# Hard queries (from our previous analysis – truly hard, all setups fail)
HARD_EXAMPLES = [
    # (subset, qi, description, key_category_terms)
    (1, 6, "Flora of the Caribbean and of Namibia",
     ["Caribbean", "Namibia"]),
    (1, 47, "Rosid genera that are also ornamental trees",
     ["Rosid", "ornamental"]),
    (1, 55, "Monotypic Zingiberales genera",
     ["Monotypic", "Zingiberales"]),
    (1, 33, "Fauna of South India excluding Holarctic birds",
     ["South India", "Holarctic"]),
    (1, 69, "Birds of Oceania that are also Passeri excluding Birds of Melanesia",
     ["Oceania", "Passeri", "Melanesia"]),
    (1, 80, "Malpighiales genera and Paleotropical flora but not Monotypic angiosperm genera",
     ["Malpighiales", "Paleotropical", "Monotypic"]),
]

for subset, qi, desc, terms in HARD_EXAMPLES:
    queries = load_queries(subset)
    docs_corpus = load_documents(subset)
    metadata = load_metadata(subset)

    q = queries[qi]
    gold_docs = q["docs"]

    print(f"\n{'─'*80}")
    print(f"Query [{subset}:{qi}]: \"{q['query']}\"")
    print(f"  original: \"{q['original_query']}\"")
    print(f"  domain: {q['metadata'].get('domain')}, #gold: {len(gold_docs)}")
    print(f"  Category terms to find: {terms}")

    text_hits = {t: 0 for t in terms}
    meta_hits = {t: 0 for t in terms}
    n_checked = 0

    for doc_title in gold_docs[:8]:
        doc_text = docs_corpus.get(doc_title, "")
        _, facets = find_entity(metadata, doc_title)
        meta_vals = get_all_metadata_values(facets) if facets else set()

        n_checked += 1
        for t in terms:
            if term_in_text(t, doc_text):
                text_hits[t] += 1
            if term_in_metadata(t, meta_vals):
                meta_hits[t] += 1

    print(f"\n  Results (checked {n_checked}/{len(gold_docs)} gold docs):")
    print(f"  {'Term':<25} {'In TEXT':>10} {'In METADATA':>12}")
    for t in terms:
        print(f"  {t:<25} {text_hits[t]:>5}/{n_checked}    {meta_hits[t]:>5}/{n_checked}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: Easy query examples (films/books) for contrast
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 90)
print("PART 2: Easy query examples — same check on film/book queries where metadata works")
print("=" * 90)

# Find easy queries: colbert recall >= 0.8
subset = 1
queries_1 = load_queries(1)
colbert_1 = load_jsonl(EVAL_DIR / "colbert_subset_1.jsonl")
docs_corpus_1 = load_documents(1)
metadata_1 = load_metadata(1)

EASY_EXAMPLES = []
for qi, rec in enumerate(colbert_1):
    if rec["recall@20"] >= 0.9 and qi < len(queries_1):
        q = queries_1[qi]
        domain = q["metadata"].get("domain", "")
        marks = extract_mark_terms(q["original_query"])
        if domain in ("films", "books") and marks and len(q["docs"]) >= 3:
            EASY_EXAMPLES.append((qi, q, marks))
    if len(EASY_EXAMPLES) >= 6:
        break

for qi, q, marks in EASY_EXAMPLES:
    gold_docs = q["docs"]
    terms = []
    for m in marks:
        # Extract key terms from each mark
        for word in m.split():
            if len(word) > 3 and word.lower() not in {"that", "also", "from", "with", "both", "films", "books", "film", "book", "the"}:
                terms.append(word)
    terms = terms[:4]

    text_hits = {t: 0 for t in terms}
    meta_hits = {t: 0 for t in terms}
    n_checked = 0

    for doc_title in gold_docs[:8]:
        doc_text = docs_corpus_1.get(doc_title, "")
        _, facets = find_entity(metadata_1, doc_title)
        meta_vals = get_all_metadata_values(facets) if facets else set()

        n_checked += 1
        for t in terms:
            if term_in_text(t, doc_text):
                text_hits[t] += 1
            if term_in_metadata(t, meta_vals):
                meta_hits[t] += 1

    print(f"\n{'─'*80}")
    print(f"Easy Query [{1}:{qi}]: \"{q['query']}\"")
    print(f"  original: \"{q['original_query']}\"")
    print(f"  domain: {q['metadata'].get('domain')}, #gold: {len(gold_docs)}")
    print(f"  Category terms: {terms}")
    print(f"\n  Results (checked {n_checked}/{len(gold_docs)} gold docs):")
    print(f"  {'Term':<25} {'In TEXT':>10} {'In METADATA':>12}")
    for t in terms:
        print(f"  {t:<25} {text_hits[t]:>5}/{n_checked}    {meta_hits[t]:>5}/{n_checked}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: Systematic term coverage across all queries by domain
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 90)
print("PART 3: Systematic term coverage — for ALL queries, per domain")
print("  For each query, extract <mark> terms, check what % of gold docs have them in text vs metadata")
print("=" * 90)

domain_stats = defaultdict(lambda: {"text_rate": [], "meta_rate": [], "n_queries": 0})

for subset in SUBSETS:
    queries_s = load_queries(subset)
    docs_corpus_s = load_documents(subset)
    metadata_s = load_metadata(subset)

    for qi, q in enumerate(queries_s):
        domain = q["metadata"].get("domain", "unknown")
        marks = extract_mark_terms(q["original_query"])
        if not marks:
            continue

        # Extract meaningful terms from marks
        terms = []
        for m in marks:
            # Use multi-word marks as-is, plus individual significant words
            terms.append(m)
            for word in m.split():
                if len(word) > 4 and word.lower() not in {"films", "books", "flora", "fauna", "birds", "that", "also", "which"}:
                    terms.append(word)
        terms = list(set(terms))[:6]  # deduplicate, limit
        if not terms:
            continue

        gold_docs = q["docs"]
        text_rates_per_term = []
        meta_rates_per_term = []

        for t in terms:
            text_count = 0
            meta_count = 0
            n_checked = 0
            for doc_title in gold_docs[:10]:
                doc_text = docs_corpus_s.get(doc_title, "")
                _, facets = find_entity(metadata_s, doc_title)
                meta_vals = get_all_metadata_values(facets) if facets else set()
                n_checked += 1
                if term_in_text(t, doc_text):
                    text_count += 1
                if term_in_metadata(t, meta_vals):
                    meta_count += 1
            if n_checked > 0:
                text_rates_per_term.append(text_count / n_checked)
                meta_rates_per_term.append(meta_count / n_checked)

        if text_rates_per_term:
            avg_text = sum(text_rates_per_term) / len(text_rates_per_term)
            avg_meta = sum(meta_rates_per_term) / len(meta_rates_per_term)
            domain_stats[domain]["text_rate"].append(avg_text)
            domain_stats[domain]["meta_rate"].append(avg_meta)
            domain_stats[domain]["n_queries"] += 1

print(f"\n  {'Domain':<12} {'#Queries':>10} {'Avg text coverage':>20} {'Avg meta coverage':>20} {'Gap':>10}")
print(f"  {'─'*72}")
for domain in ["films", "books", "animals", "plants"]:
    s = domain_stats[domain]
    n = s["n_queries"]
    avg_t = sum(s["text_rate"]) / n if n else 0
    avg_m = sum(s["meta_rate"]) / n if n else 0
    gap = avg_m - avg_t
    print(f"  {domain:<12} {n:>10} {avg_t:>18.1%} {avg_m:>18.1%} {gap:>+9.1%}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: Retrieval performance by domain
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 90)
print("PART 4: Retrieval performance by domain — colbert vs meta_colbert vs meta_rerank")
print("=" * 90)

domain_recalls = defaultdict(lambda: defaultdict(list))
setups_to_check = ["colbert", "meta_colbert", "meta_rerank", "dense", "meta_dense"]

for subset in SUBSETS:
    queries_s = load_queries(subset)
    per_query_by_setup = {}
    for setup in setups_to_check:
        per_query_by_setup[setup] = load_jsonl(EVAL_DIR / f"{setup}_subset_{subset}.jsonl")

    for qi, q in enumerate(queries_s):
        domain = q["metadata"].get("domain", "unknown")
        for setup in setups_to_check:
            rec = per_query_by_setup[setup][qi]
            domain_recalls[domain][setup].append(rec["recall@20"])

print(f"\n  {'Domain':<12}", end="")
for s in setups_to_check:
    print(f"  {s:>14}", end="")
print(f"  {'meta_colbert - colbert':>24}")
print(f"  {'─'*100}")

for domain in ["films", "books", "animals", "plants"]:
    print(f"  {domain:<12}", end="")
    vals = {}
    for s in setups_to_check:
        v = domain_recalls[domain][s]
        avg_v = sum(v) / len(v) if v else 0
        vals[s] = avg_v
        print(f"  {avg_v:>14.4f}", end="")
    delta = vals.get("meta_colbert", 0) - vals.get("colbert", 0)
    print(f"  {delta:>+23.4f}")

# Per-domain, count how often meta_colbert > colbert
print(f"\n  Per-query improvement rates (meta_colbert recall > colbert recall):")
for domain in ["films", "books", "animals", "plants"]:
    n = len(domain_recalls[domain]["colbert"])
    better = sum(1 for i in range(n)
                 if domain_recalls[domain]["meta_colbert"][i] > domain_recalls[domain]["colbert"][i] + 0.01)
    worse = sum(1 for i in range(n)
                if domain_recalls[domain]["meta_colbert"][i] < domain_recalls[domain]["colbert"][i] - 0.01)
    same = n - better - worse
    print(f"  {domain:<12} better: {better:>3}/{n} ({100*better/n:.0f}%)  worse: {worse:>3}/{n} ({100*worse/n:.0f}%)  same: {same:>3}/{n} ({100*same/n:.0f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# PART 5: For the "easy" domains (films/books) — WHERE does metadata help?
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 90)
print("PART 5: Within films/books — when does meta_colbert beat colbert?")
print("=" * 90)

for domain_focus in ["films", "books"]:
    print(f"\n{'─'*40} {domain_focus} {'─'*40}")
    
    # Collect queries where meta_colbert > colbert
    helps = []
    hurts = []
    
    for subset in SUBSETS:
        queries_s = load_queries(subset)
        colbert_pq = load_jsonl(EVAL_DIR / f"colbert_subset_{subset}.jsonl")
        mc_pq = load_jsonl(EVAL_DIR / f"meta_colbert_subset_{subset}.jsonl")
        mr_pq = load_jsonl(EVAL_DIR / f"meta_rerank_subset_{subset}.jsonl")
        meta_res = []
        try:
            with open(META_DIR / f"quest_eval_results_val_expanded_subset_{subset}.jsonl") as f:
                meta_res = [json.loads(l) for l in f]
        except:
            pass
        
        for qi, q in enumerate(queries_s):
            if q["metadata"].get("domain") != domain_focus:
                continue
            c_r = colbert_pq[qi]["recall@20"]
            mc_r = mc_pq[qi]["recall@20"]
            mr_r = mr_pq[qi]["recall@20"]
            delta = mc_r - c_r
            n_filt = meta_res[qi].get("n_filtered_docs", "?") if qi < len(meta_res) else "?"
            
            entry = {
                "subset": subset, "qi": qi,
                "query": q["query"],
                "original": q["original_query"],
                "template": q["metadata"].get("template", "?"),
                "n_gold": len(q["docs"]),
                "colbert": c_r, "meta_colbert": mc_r, "meta_rerank": mr_r,
                "delta": delta,
                "n_filtered": n_filt,
            }
            if delta > 0.05:
                helps.append(entry)
            elif delta < -0.05:
                hurts.append(entry)
    
    helps.sort(key=lambda x: -x["delta"])
    hurts.sort(key=lambda x: x["delta"])
    
    print(f"  meta_colbert helps (>5% recall gain): {len(helps)}")
    for e in helps[:5]:
        print(f"    [{e['subset']}:{e['qi']}] \"{e['query'][:70]}\"")
        print(f"      template: {e['template']} | gold: {e['n_gold']} | c: {e['colbert']:.3f} → mc: {e['meta_colbert']:.3f} ({e['delta']:+.3f}) | n_filt: {e['n_filtered']}")
    
    print(f"\n  meta_colbert hurts (>5% recall loss): {len(hurts)}")
    for e in hurts[:5]:
        print(f"    [{e['subset']}:{e['qi']}] \"{e['query'][:70]}\"")
        print(f"      template: {e['template']} | gold: {e['n_gold']} | c: {e['colbert']:.3f} → mc: {e['meta_colbert']:.3f} ({e['delta']:+.3f}) | n_filt: {e['n_filtered']}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 6: Query complexity vs metadata benefit
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 90)
print("PART 6: Template complexity vs metadata benefit")
print("=" * 90)

template_stats = defaultdict(lambda: {"deltas": [], "colbert": [], "meta_colbert": []})

for subset in SUBSETS:
    queries_s = load_queries(subset)
    colbert_pq = load_jsonl(EVAL_DIR / f"colbert_subset_{subset}.jsonl")
    mc_pq = load_jsonl(EVAL_DIR / f"meta_colbert_subset_{subset}.jsonl")
    
    for qi, q in enumerate(queries_s):
        template = q["metadata"].get("template", "?")
        c_r = colbert_pq[qi]["recall@20"]
        mc_r = mc_pq[qi]["recall@20"]
        template_stats[template]["deltas"].append(mc_r - c_r)
        template_stats[template]["colbert"].append(c_r)
        template_stats[template]["meta_colbert"].append(mc_r)

print(f"\n  {'Template':<40} {'N':>5} {'Colbert':>10} {'Meta-Colbert':>14} {'Delta':>10}")
print(f"  {'─'*80}")
for t, s in sorted(template_stats.items(), key=lambda x: -len(x[1]["deltas"])):
    n = len(s["deltas"])
    avg_c = sum(s["colbert"]) / n
    avg_mc = sum(s["meta_colbert"]) / n
    avg_d = sum(s["deltas"]) / n
    print(f"  {t:<40} {n:>5} {avg_c:>10.4f} {avg_mc:>14.4f} {avg_d:>+10.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 7: Concrete text excerpt examples
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 90)
print("PART 7: Concrete text excerpts — what gold doc text actually says")
print("  (Showing first 300 chars of text for hard vs easy gold docs)")
print("=" * 90)

subset = 1
queries_1 = load_queries(1)
docs_corpus_1 = load_documents(1)

# Hard example: Query 55 "Monotypic Zingiberales genera"
print(f"\n{'─'*80}")
q = queries_1[55]
print(f"HARD: Query 55: \"{q['query']}\"")
print(f"  original: \"{q['original_query']}\"")
print(f"  Terms to match: 'Monotypic', 'Zingiberales'")
for doc_title in q["docs"][:3]:
    text = docs_corpus_1.get(doc_title, "NOT FOUND")
    print(f"\n  Gold doc: \"{doc_title}\"")
    print(f"  Text (first 400 chars):")
    print(f"    {text[:400]}...")
    # Search for key terms
    for t in ["Monotypic", "Zingiberales", "monocot", "ginger"]:
        if t.lower() in text.lower():
            # Find context
            idx = text.lower().index(t.lower())
            context = text[max(0, idx-50):idx+len(t)+50]
            print(f"  ✓ Found '{t}' in text: \"...{context}...\"")
        else:
            print(f"  ✗ '{t}' NOT in text")

# Hard example: Query 47 "Rosid genera that are also ornamental trees"
print(f"\n{'─'*80}")
q = queries_1[47]
print(f"HARD: Query 47: \"{q['query']}\"")
print(f"  original: \"{q['original_query']}\"")
for doc_title in q["docs"][:3]:
    text = docs_corpus_1.get(doc_title, "NOT FOUND")
    print(f"\n  Gold doc: \"{doc_title}\"")
    print(f"  Text (first 400 chars):")
    print(f"    {text[:400]}...")
    for t in ["Rosid", "rosid", "ornamental", "garden", "decorat"]:
        if t.lower() in text.lower():
            idx = text.lower().index(t.lower())
            context = text[max(0, idx-50):idx+len(t)+50]
            print(f"  ✓ Found '{t}' in text: \"...{context}...\"")
        else:
            print(f"  ✗ '{t}' NOT in text")

# Easy example: find a film query with good metadata
colbert_1 = load_jsonl(EVAL_DIR / "colbert_subset_1.jsonl")
for qi, rec in enumerate(colbert_1):
    if rec["recall@20"] >= 0.9 and qi < len(queries_1):
        q = queries_1[qi]
        if q["metadata"].get("domain") == "films" and len(extract_mark_terms(q["original_query"])) >= 2:
            print(f"\n{'─'*80}")
            print(f"EASY: Query {qi}: \"{q['query']}\"")
            print(f"  original: \"{q['original_query']}\"")
            marks = extract_mark_terms(q["original_query"])
            print(f"  Mark terms: {marks}")
            for doc_title in q["docs"][:2]:
                text = docs_corpus_1.get(doc_title, "NOT FOUND")
                print(f"\n  Gold doc: \"{doc_title}\"")
                print(f"  Text (first 400 chars):")
                print(f"    {text[:400]}...")
                for m in marks:
                    for word in m.split():
                        if len(word) > 3 and word.lower() not in {"that", "also", "films", "the"}:
                            if word.lower() in text.lower():
                                print(f"  ✓ Found '{word}' in text")
                            else:
                                print(f"  ✗ '{word}' NOT in text")
            break
