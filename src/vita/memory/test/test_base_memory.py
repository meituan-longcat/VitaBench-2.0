"""
Tests for BaseMemory interface and tool isolation.

Covers:
- Abstract method contracts (cannot instantiate BaseMemory directly)
- Constructor defaults (language, top_k, similarity_threshold)
- Tool isolation: memory subclass tools vs ToolKitBase domain tools
"""

import pytest

from vita.memory.base import BaseMemory


class TestBaseMemoryAbstract:

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseMemory(language="chinese")

    def test_subclass_must_implement_read_update(self):
        class IncompleteMemory(BaseMemory):
            pass

        with pytest.raises(TypeError):
            IncompleteMemory(language="chinese")


class TestBaseMemorySubclass:

    def _make_memory_class(self):
        class SimpleMemory(BaseMemory):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._data = ""

            def read(self, query=None):
                return self._data or "empty"

            def update(self, new_interactions, llm=None, llm_args=None, **kwargs):
                self._data += str(new_interactions)
                return self._data

        return SimpleMemory

    def test_default_params(self):
        cls = self._make_memory_class()
        mem = cls(language="chinese")
        assert mem.language == "chinese"
        assert mem.top_k == 5
        assert mem.similarity_threshold == 0.0

    def test_custom_params(self):
        cls = self._make_memory_class()
        mem = cls(language="english", top_k=10, similarity_threshold=0.3)
        assert mem.language == "english"
        assert mem.top_k == 10
        assert mem.similarity_threshold == 0.3

    def test_reset_default_noop(self):
        cls = self._make_memory_class()
        mem = cls(language="chinese")
        mem.update([{"type": "order", "timestamp": "", "content": "test"}])
        mem.reset()  # default BaseMemory.reset is a no-op
        # Data is still there because SimpleMemory doesn't override reset
        assert "test" in mem.read()

    def test_no_custom_tools_by_default(self):
        cls = self._make_memory_class()
        mem = cls(language="chinese")
        # A plain subclass with no @is_tool methods should have no memory tools
        assert len(mem.tools) == 0

    def test_get_tools_returns_dict(self):
        cls = self._make_memory_class()
        mem = cls(language="chinese")
        tools = mem.get_tools()
        assert isinstance(tools, dict)


class TestToolIsolation:
    """Memory subclasses should not expose ToolKitBase's domain tools."""

    def test_rewrite_memory_no_domain_tools(self):
        from vita.memory.rewrite_memory import RewriteMemory
        mem = RewriteMemory(language="chinese")
        tool_names = set(mem.tools.keys())
        # Should not contain ToolKitBase domain tools like 'get_weather', 'get_nearby', etc.
        domain_tools = {"get_weather", "get_nearby_services", "get_nearby"}
        assert tool_names.isdisjoint(domain_tools)

    def test_rag_memory_no_domain_tools(self):
        from vita.memory.rag_memory import RAGMemory
        mem = RAGMemory(language="chinese")
        tool_names = set(mem.tools.keys())
        domain_tools = {"get_weather", "get_nearby_services", "get_nearby"}
        assert tool_names.isdisjoint(domain_tools)
