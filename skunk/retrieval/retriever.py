from __future__ import annotations

import json
import re

from skunk.retrieval.nodes import PlotNode, TableNode, TextNode
from skunk.retrieval.nodes.document_node import DocumentNode
from skunk.retrieval.nodes.llm_wrapper import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_WRAPPER,
    parse_json_response,
)

KEYWORD_EXTRACTION_LIMIT = 10
KEYWORD_EXPANSION_LIMIT = 3
MAX_NODE_CONTENT_CHARS = 1200


class Retriever:
    def __init__(
        self,
        index,
        candidate_nodes_per_keyword: int = 20,
        relevant_nodes_k: int = 10,
        final_documents_k: int = 5,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        use_cache: bool | None = None,
    ):
        if candidate_nodes_per_keyword <= 0:
            raise ValueError("candidate_nodes_per_keyword must be positive")
        if relevant_nodes_k <= 0:
            raise ValueError("relevant_nodes_k must be positive")
        if final_documents_k <= 0:
            raise ValueError("final_documents_k must be positive")

        self.index = index
        self.candidate_nodes_per_keyword = candidate_nodes_per_keyword
        self.relevant_nodes_k = relevant_nodes_k
        self.final_documents_k = final_documents_k
        self.embedding_model = embedding_model
        self.use_cache = getattr(index, "use_cache", True) if use_cache is None else use_cache

    def retrieve(self, question: str) -> list[DocumentNode]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        assert self.index.initialized, "SemanticDocumentIndex must be initialized before retrieval!"
        content_records = self._content_node_records()
        content_records_by_id = {record["node_id"]: record for record in content_records}
        if not content_records:
            return []

        keywords = self.decompose_query(question)
        query_terms = []
        seen_terms = set()
        for term in [question, *keywords]:
            normalized = re.sub(r"\s+", " ", term.strip().lower())
            if normalized and normalized not in seen_terms:
                query_terms.append(term.strip())
                seen_terms.add(normalized)

        term_embeddings = DEFAULT_LLM_WRAPPER.embed_texts(
            query_terms,
            model=self.embedding_model,
            use_cache=self.use_cache,
        )
        candidates_by_id = {}
        for term, term_embedding in zip(query_terms, term_embeddings, strict=True):
            scored_records = []
            for matched_node_id, score in self.index.search_description_embeddings(
                term_embedding,
                self.candidate_nodes_per_keyword,
            ):
                matched_node = self.index.get_node(matched_node_id)
                if matched_node_id in content_records_by_id:
                    scored_records.append((score, content_records_by_id[matched_node_id]))
                elif hasattr(matched_node, "page_id"):
                    scored_records.extend(
                        (score, record) for record in content_records if record["page_id"] == matched_node.page_id
                    )
                elif hasattr(matched_node, "document_id"):
                    scored_records.extend(
                        (score, record)
                        for record in content_records
                        if record["document_id"] == matched_node.document_id
                    )

            scored_records.sort(key=lambda scored_record: scored_record[0], reverse=True)
            for score, record in scored_records[: self.candidate_nodes_per_keyword]:
                node_id = record["node_id"]
                if node_id not in candidates_by_id:
                    candidates_by_id[node_id] = {
                        **record,
                        "score": score,
                        "matched_terms": [term],
                    }
                else:
                    candidates_by_id[node_id]["score"] = max(candidates_by_id[node_id]["score"], score)
                    candidates_by_id[node_id]["matched_terms"].append(term)

        candidates = sorted(candidates_by_id.values(), key=lambda record: record["score"], reverse=True)
        relevant_node_ids = self.judge_relevant_nodes(question, candidates)
        if not relevant_node_ids:
            relevant_node_ids = [record["node_id"] for record in candidates[: self.relevant_nodes_k]]

        document_scores = {}
        for rank, node_id in enumerate(relevant_node_ids):
            if node_id not in candidates_by_id:
                continue
            record = candidates_by_id[node_id]
            document_id = record["document_id"]
            rank_bonus = self.relevant_nodes_k - rank
            document_scores[document_id] = document_scores.get(document_id, 0.0) + record["score"] + rank_bonus

        ranked_document_ids = sorted(document_scores, key=lambda document_id: document_scores[document_id], reverse=True)
        return [self.index.documents[document_id] for document_id in ranked_document_ids[: self.final_documents_k]]

    def decompose_query(self, question: str) -> list[str]:
        tree_context = self.explore_tree(max_lines=80)
        prompt = f"""
You are preparing a document tree retrieval query.

Use the available tree exploration context and the user's question to extract keywords that are likely to route retrieval to answer-containing nodes. Expand each keyword with closely related terms, abbreviations, and alternate phrasings that may appear in document descriptions.

Rules:
- Return only valid JSON.
- Include at most {KEYWORD_EXTRACTION_LIMIT} primary keywords.
- Include at most {KEYWORD_EXPANSION_LIMIT} expansions per primary keyword.
- Prefer specific entities, dates, measures, table labels, and domain phrases.
- Do not include generic words such as document, page, answer, information, or question.

JSON shape:
{{
  "keywords": [
    {{
      "keyword": "primary retrieval keyword",
      "expansions": ["related keyword"]
    }}
  ]
}}

Question:
{question}

Tree exploration context:
{tree_context}
""".strip()
        try:
            parsed = parse_json_response(DEFAULT_LLM_WRAPPER.call_llm(prompt, use_cache=self.use_cache))
        except Exception:
            parsed = {"keywords": []}

        keywords = []
        seen_keywords = set()
        for item in parsed.get("keywords", []):
            terms = [item] if isinstance(item, str) else [item.get("keyword", ""), *item.get("expansions", [])]
            for term in terms:
                normalized = re.sub(r"\s+", " ", str(term).strip().lower())
                if normalized and normalized not in seen_keywords:
                    keywords.append(str(term).strip())
                    seen_keywords.add(normalized)

        if keywords:
            return keywords

        fallback_terms = re.findall(r"[A-Za-z][A-Za-z0-9$%.-]{2,}", question)
        for term in fallback_terms:
            normalized = term.lower()
            if normalized not in seen_keywords:
                keywords.append(term)
                seen_keywords.add(normalized)
        return keywords[:KEYWORD_EXTRACTION_LIMIT]

    def explore_tree(self, max_lines: int = 80) -> str:
        lines = []
        for document in self.index.documents.values():
            document_title = document.document_title or document.filename or document.document_id
            lines.append(f"DOCUMENT {document.document_id}: {document_title}; {document.description}")
            for page in document.page_nodes:
                lines.append(f"PAGE {page.page_id}: page_number={page.page_number}; {page.description}")
                for text_node in page.text_nodes:
                    lines.append(f"TEXT {text_node.text_id}: {text_node.description}")
                for table_node in page.table_nodes:
                    table_title = table_node.table_title or "Untitled table"
                    lines.append(f"TABLE {table_node.table_id}: {table_title}; {table_node.description}")
                for plot_node in page.plot_nodes:
                    plot_title = plot_node.plot_title or "Untitled plot"
                    lines.append(f"PLOT {plot_node.plot_id}: {plot_title}; {plot_node.description}")
                if len(lines) >= max_lines:
                    return "\n".join(lines[:max_lines])
        return "\n".join(lines[:max_lines])

    def judge_relevant_nodes(self, question: str, candidates: list[dict]) -> list[str]:
        prompt_candidates = []
        for candidate in candidates:
            node = candidate["node"]
            content = self._node_content(node)
            prompt_candidates.append(
                {
                    "node_id": candidate["node_id"],
                    "document_id": candidate["document_id"],
                    "document_title": candidate["document_title"],
                    "page_id": candidate["page_id"],
                    "page_number": candidate["page_number"],
                    "node_type": candidate["node_type"],
                    "description": getattr(node, "description", ""),
                    "content": content[:MAX_NODE_CONTENT_CHARS],
                    "matched_terms": candidate["matched_terms"],
                    "retrieval_score": round(candidate["score"], 6),
                }
            )

        prompt = f"""
You are judging whether retrieved semantic index nodes contain information needed to answer a user's question.

Rules:
- Return only valid JSON.
- Select at most {self.relevant_nodes_k} node IDs.
- Select nodes that are directly relevant to answering the question or likely contain the answer.
- Prefer nodes with concrete facts, tables, figures, quantities, dates, or named entities needed by the question.
- Do not select nodes that only match vague topic words.

JSON shape:
{{
  "relevant_nodes": [
    {{
      "node_id": "node id from candidates",
      "reason": "short reason"
    }}
  ]
}}

Question:
{question}

Candidate nodes:
{json.dumps(prompt_candidates, indent=2)}
""".strip()
        try:
            parsed = parse_json_response(
                DEFAULT_LLM_WRAPPER.call_llm(
                    prompt,
                    max_tokens=4096,
                    use_cache=self.use_cache,
                )
            )
        except Exception:
            return []

        candidate_ids = {candidate["node_id"] for candidate in candidates}
        relevant_node_ids = []
        for item in parsed.get("relevant_nodes", []):
            node_id = item if isinstance(item, str) else item.get("node_id", "")
            if node_id in candidate_ids and node_id not in relevant_node_ids:
                relevant_node_ids.append(node_id)
            if len(relevant_node_ids) >= self.relevant_nodes_k:
                break
        return relevant_node_ids

    def _content_node_records(self) -> list[dict]:
        records = []
        for document in self.index.documents.values():
            for page in document.page_nodes:
                for node_type, nodes in (
                    ("text", page.text_nodes),
                    ("table", page.table_nodes),
                    ("plot", page.plot_nodes),
                ):
                    for node in nodes:
                        records.append(
                            {
                                "node": node,
                                "node_id": self.index.get_node_id(node),
                                "node_type": node_type,
                                "page_id": page.page_id,
                                "page_number": page.page_number,
                                "document_id": document.document_id,
                                "document_title": document.document_title,
                            }
                        )
        return records

    def _node_content(self, node: TextNode | TableNode | PlotNode) -> str:
        if isinstance(node, TextNode):
            return node.text
        if isinstance(node, TableNode):
            return f"{node.table_title}\n{node.table_data}".strip()
        return f"{node.plot_title}\n{node.description}".strip()
