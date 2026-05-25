import os
import shlex
import subprocess

from chromadb.api.models.Collection import Collection
from google import genai

# Tokens that are allowed to appear as the *command name* in each segment of a
# grep pipeline. Anything else causes the command to be rejected.
GREP_ALLOWED_COMMANDS = {"grep", "xargs", "head", "tail", "ls", "cat"}

# Any of these substrings, if present anywhere in a grep command, cause it to
# be rejected outright -- even if the leading command token is allowed.
GREP_FORBIDDEN_PATTERNS = [
    "rm", "mv", "cp", "dd", "mkfifo", "mkdir", "rmdir", "chmod", "chown",
    "ln ", "touch", "tee", "sudo", "su ", "kill", "pkill", "shutdown",
    "reboot", "curl", "wget", "ssh", "scp", "rsync", "ftp", "nc ", "ncat",
    "python", "node", "perl", "ruby", " sh ", "bash", "zsh", "exec",
    "eval", "source", "$(", "`", ">", "<", "&", ";",
]

def _make_retrieve_page_info(clean_page_map: dict[str, list]):
    def retrieve_page_info(year_month_page_tuples):
        chunks = []
        for item in year_month_page_tuples:
            year, month, page_id = item
            key = f"{int(year)}-{int(month):02d}-{int(page_id)}"
            entry = clean_page_map.get(key)
            if entry is None:
                chunks.append(f"=== {key} ===\n[no such page (or no content on page)]")
                continue
            filepath = entry[0]
            try:
                with open(filepath) as f:
                    text = f.read()
            except OSError as e:
                chunks.append(f"=== {key} ===\n[error reading file: {e}]")
                continue
            chunks.append(f"=== {key} ({filepath}) ===\n{text}")
        return "\n\n".join(chunks)

    return retrieve_page_info


def _make_vector_search(chroma_collection: Collection, emb_model_id: str, genai_client: genai.Client):
    def _build_where(
        filter_year_months: list[tuple[int, int]] | None,
        filter_page_ids: list[int] | None,
    ) -> dict | None:
        """Build a ChromaDB `where` filter from the optional year-month and page-id constraints.

        Metadata schema:
          year     – zero-padded string e.g. "1941"
          month    – zero-padded string e.g. "01"
          page_id  – int
        """
        clauses: list[dict] = []

        if filter_year_months:
            ym_conditions = [
                {"$and": [{"year": str(y)}, {"month": f"{m:02d}"}]}
                for y, m in filter_year_months
            ]
            # $or requires at least two items; use the condition directly for a single entry.
            if len(ym_conditions) == 1:
                clauses.append(ym_conditions[0])
            else:
                clauses.append({"$or": ym_conditions})

        if filter_page_ids:
            if len(filter_page_ids) == 1:
                clauses.append({"page_id": filter_page_ids[0]})
            else:
                clauses.append({"page_id": {"$in": filter_page_ids}})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def vector_search(
        query: str,
        top_k: int,
        filter_year_months: list[tuple[int, int]] | None = None,
        filter_page_ids: list[int] | None = None,
    ) -> str:
        # embed the query using the same model that produced the stored embeddings.
        emb_result = genai_client.models.embed_content(model=emb_model_id, contents=query)
        query_embedding = list(emb_result.embeddings[0].values)  # type: ignore

        where = _build_where(filter_year_months, filter_page_ids)
        query_kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["metadatas", "distances"],
        }
        if where is not None:
            query_kwargs["where"] = where

        try:
            results = chroma_collection.query(**query_kwargs)
        except Exception as e:
            return f"[vector_search error: {e}]"

        metadatas = results["metadatas"][0]  # type: ignore
        distances = results["distances"][0]  # type: ignore

        chunks = []
        for rank, (meta, dist) in enumerate(zip(metadatas, distances, strict=True), 1):
            page_key = meta.get("page_key", "?")
            elt_type = meta.get("type", "?")
            cleaned = meta.get("cleaned", "")
            chunks.append(
                f"[{rank}] page_key={page_key} | type={elt_type} | distance={dist:.4f}\n{cleaned}"
            )
        return "\n\n".join(chunks)

    return vector_search


def _validate_grep_cmd(grep_cmd: str) -> None:
    if not isinstance(grep_cmd, str) or not grep_cmd.strip():
        raise ValueError("grep_cmd must be a non-empty string")

    for pat in GREP_FORBIDDEN_PATTERNS:
        if pat in grep_cmd:
            raise ValueError(
                f"grep_cmd rejected: contains forbidden token {pat!r}"
            )

    # Split on pipe characters that are NOT inside quotes by tokenizing first.
    try:
        all_tokens = shlex.split(grep_cmd)
    except ValueError as e:
        raise ValueError(f"grep_cmd is not valid shell syntax: {e}") from e

    # Reconstruct pipeline segments by splitting on bare "|" tokens.
    segments: list[list[str]] = []
    current: list[str] = []
    for tok in all_tokens:
        if tok == "|":
            segments.append(current)
            current = []
        else:
            current.append(tok)
    segments.append(current)

    for tokens in segments:
        if not tokens:
            raise ValueError("grep_cmd contains an empty pipeline segment")
        leading = os.path.basename(tokens[0])
        if leading not in GREP_ALLOWED_COMMANDS:
            raise ValueError(
                f"grep_cmd rejected: leading command {leading!r} is not in "
                f"the allowlist {sorted(GREP_ALLOWED_COMMANDS)}"
            )


def run_grep(grep_cmd: str) -> str:
    _validate_grep_cmd(grep_cmd)
    try:
        proc = subprocess.run(
            grep_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return "[grep timed out after 30s]"
    out = proc.stdout
    if proc.returncode not in (0, 1):  # grep returns 1 when no matches found
        out += f"\n[stderr]\n{proc.stderr}"
    return out

def final_answer(page_keys):
    return page_keys