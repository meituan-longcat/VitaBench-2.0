"""
NullMemory: A no-op memory backend for baseline experiments.

Returns empty strings, stores nothing, exposes no tools.
Use this to measure the impact of memory on agent performance.

Usage:
    vita run --domain personalization --memory-type null
"""

from typing import Optional

from vita.memory.base import BaseMemory


class NullMemory(BaseMemory):
    """No-op memory backend.

    read() returns empty string (system prompt memory section will be blank).
    update() does nothing.
    No @is_tool methods: agent gets no memory tools.

    Useful as a baseline to isolate the effect of memory on agent behavior.
    """

    def __init__(self, language: str = None, **kwargs):
        super().__init__(language=language, **kwargs)

    def read(self, query: str = None) -> str:
        """Return empty string. No memory content injected into system prompt."""
        return ""

    def update(
        self,
        new_interactions: list,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        **kwargs,
    ) -> str:
        """No-op. Interactions are discarded."""
        return ""

    def reset(self):
        """No-op. Nothing to reset."""
        pass
