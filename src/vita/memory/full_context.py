"""
FullContextMemory: A no-management memory backend that concatenates all interactions.

No LLM calls, no summarization, no retrieval logic.
- update(): formats and appends new interactions to the full text
- read(): returns the full text, truncated from the head to fit max_tokens

Usage:
    vita run --domain personalization --memory-type full_context

Token counting uses tiktoken's cl100k_base encoding (gpt-4 / gpt-4.1 / gpt-3.5).
Because the server has no outbound network, the encoding must be preloaded:

    export TIKTOKEN_CACHE_DIR=<repo>/vita-bench-personalize/data/tiktoken_cache

That directory ships with the sha1-named encoding files. If tiktoken is
unavailable or the cache is missing, we fall back to a chars/token heuristic.
"""

import os
from typing import Optional

from vita.memory.base import BaseMemory
from vita.memory.rewrite_memory import RewriteMemory

# Default location for the preloaded tiktoken cache, relative to this repo.
# Files must be named as the sha1 hex of their download URL (tiktoken's
# convention). Users can override via the TIKTOKEN_CACHE_DIR env var.
_DEFAULT_TIKTOKEN_CACHE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "tiktoken_cache")
)
if not os.environ.get("TIKTOKEN_CACHE_DIR") and os.path.isdir(_DEFAULT_TIKTOKEN_CACHE):
    os.environ["TIKTOKEN_CACHE_DIR"] = _DEFAULT_TIKTOKEN_CACHE

try:
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception as _e:  # tiktoken missing, or cache not populated
    _ENCODING = None
    _TIKTOKEN_INIT_ERROR = _e

# Fallback ratio when tiktoken is unavailable. cl100k averages ~2.5 chars/token
# for mixed Chinese text; 2 keeps us within budget.
_CHARS_PER_TOKEN = 2


class FullContextMemory(BaseMemory):
    """Append-only full context memory with token-based truncation.

    Accumulates all interactions as formatted text. On read(), if the token
    count exceeds max_tokens, the oldest content (head) is removed so only
    the most recent max_tokens tokens are returned. Truncation happens at a
    token boundary; the first (possibly partial) line after truncation is
    dropped so the returned text starts at a line boundary.
    """

    def __init__(
        self,
        language: str = None,
        max_tokens: int = 4096,
        **kwargs,
    ):
        super().__init__(language=language, **kwargs)
        self.max_tokens = max_tokens
        self._memory_text: str = ""

    @staticmethod
    def _count_tokens(text: str) -> int:
        if _ENCODING is not None:
            return len(_ENCODING.encode(text))
        return max(1, len(text) // _CHARS_PER_TOKEN)

    def read(self, query: str = None) -> str:
        """Return the full memory content, truncated from the head if needed."""
        if not self._memory_text:
            return ""

        if _ENCODING is not None:
            token_ids = _ENCODING.encode(self._memory_text)
            if len(token_ids) <= self.max_tokens:
                return self._memory_text
            tail_ids = token_ids[-self.max_tokens:]
            tail = _ENCODING.decode(tail_ids)
        else:
            if self._count_tokens(self._memory_text) <= self.max_tokens:
                return self._memory_text
            max_chars = self.max_tokens * _CHARS_PER_TOKEN
            tail = self._memory_text[-max_chars:]

        # Drop a leading partial line so the output starts cleanly.
        newline_idx = tail.find("\n")
        if newline_idx != -1 and newline_idx < len(tail) - 1:
            tail = tail[newline_idx + 1:]
        return tail

    def update(
        self,
        new_interactions: list,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        **kwargs,
    ) -> str:
        """Append formatted interactions to memory. LLM params are ignored."""
        if not new_interactions:
            return self._memory_text

        interactions_text = RewriteMemory._format_interactions(new_interactions)

        if self._memory_text:
            self._memory_text = f"{self._memory_text}\n{interactions_text}"
        else:
            self._memory_text = interactions_text

        return self._memory_text

    def reset(self):
        """Reset memory to empty state."""
        self._memory_text = ""
