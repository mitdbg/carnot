"""Shared constants. Currently just the per-image token estimate; more of the
library's scattered constants will migrate here over time."""

# Flat per-image input-token estimate (~one rendered page at provider vision rates).
# Used by the TPM throttle's pre-call estimate (`llm_client._estimate_prompt_tokens`)
# and the agent's context-size tracking (`multi_turn_agent._count_tokens`) — both are
# pacing/tracking heuristics, not billing.
IMAGE_TOKENS_EST = 1000

# The house ~4 chars/token text heuristic, used (a) wherever a token budget is converted
# to a character cap with no string to tokenize — the `SemanticFilterTool` output cap and
# the semantic-filter judge's context sizing (`_judge_doc_char_budget`) — and (b) as
# `common.estimate_tokens`'s fallback for strings too large to BPE-encode. String→token
# conversion goes through `estimate_tokens`, not this constant directly. An estimate,
# not a billing figure.
CHARS_PER_TOKEN_EST = 4
