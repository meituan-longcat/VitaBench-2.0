"""
Base memory interface for the personalization domain.

BaseMemory inherits ToolKitBase so that @is_tool methods defined on
memory subclasses are auto-discovered by the ToolKitType metaclass
and can be injected into the agent's tool list by the orchestrator.

Researchers implement read() and update(), and optionally define
custom tools via @is_tool decorators. The framework handles the rest:
- read() output is injected into the agent's system prompt
- update() is called between subtasks by the orchestrator
- @is_tool methods are auto-registered into the agent's tool list
"""

from abc import abstractmethod
from typing import Callable, Dict, Optional

from vita.environment.toolkit import ToolKitBase, TOOL_ATTR
from vita.environment.tool import Tool, as_tool


# Collect ToolKitBase's own @is_tool method names so we can filter them out.
# These are domain-specific tools (weather, get_nearby, etc.) that require db.
_TOOLKITBASE_TOOL_NAMES = set()
for _name, _method in vars(ToolKitBase).items():
    if hasattr(_method, TOOL_ATTR):
        _TOOLKITBASE_TOOL_NAMES.add(_name)


class BaseMemory(ToolKitBase):
    """Abstract base class for user preference memory backends.

    Inherits ToolKitBase to enable @is_tool auto-discovery.
    Overrides get_tools() and tools property to filter out ToolKitBase's
    domain-specific tools (weather, get_nearby, etc.) which require
    self.db and would crash on memory instances.
    """

    def __init__(self, language: str = None, top_k: int = 5,
                 similarity_threshold: float = 0.0, **kwargs):
        super().__init__(db=None)
        self.language = language
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    # ── Framework-called core interface ──

    @abstractmethod
    def read(self, query: str = None) -> str:
        """Return the current memory content as a string.

        The framework injects this into the agent's system prompt
        automatically before each conversation turn.

        Args:
            query: Optional retrieve query (e.g., task instruction).
                   When provided, return only relevant memories via retrieval.
                   When None, return all memories (backward compatible).

        Returns:
            A human-readable string of the user's preference memory.
        """

    @abstractmethod
    def update(
        self,
        new_interactions: list,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        **kwargs,
    ) -> str:
        """Process new interactions and update memory.

        Called by the orchestrator between subtasks. Researchers can
        also define @is_tool WRITE methods for in-conversation updates.

        Args:
            new_interactions: List of new user interactions to incorporate.
            llm: Optional LLM model name for memory processing.
            llm_args: Optional LLM arguments.

        Returns:
            Updated memory string.
        """

    def reset(self):
        """Reset memory to empty state. Optional override."""
        pass

    # ── Tool isolation: filter out ToolKitBase's domain tools ──

    def _get_memory_tool_names(self) -> set:
        """Get tool names defined on memory classes only (not ToolKitBase)."""
        all_tool_names = set(self._func_tools.keys())
        return all_tool_names - _TOOLKITBASE_TOOL_NAMES

    @property
    def tools(self) -> Dict[str, Callable]:
        """Get only memory-defined tools, excluding ToolKitBase's domain tools."""
        memory_names = self._get_memory_tool_names()
        return {name: getattr(self, name) for name in memory_names}

    def get_tools(self) -> Dict[str, Tool]:
        """Get memory tools as Tool objects, excluding ToolKitBase's domain tools.

        Returns:
            Dictionary of Tool objects for researcher-defined @is_tool methods only.
        """
        return {name: as_tool(tool) for name, tool in self.tools.items()}
