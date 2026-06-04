"""
Tests for the memory factory: create_memory() from vita.memory.__init__.

Covers:
- Creating built-in memory types by name
- Registry completeness
- Unknown type raises ValueError
- Custom class import via dotted path
- Type validation
"""

import pytest

from vita.memory import create_memory, MEMORY_REGISTRY
from vita.memory.base import BaseMemory
from vita.memory.rewrite_memory import RewriteMemory
from vita.memory.rag_memory import RAGMemory
from vita.memory.null_memory import NullMemory
from vita.memory.full_context import FullContextMemory


class TestCreateMemoryBuiltIn:

    def test_create_rewrite(self):
        mem = create_memory(memory_type="rewrite", language="chinese")
        assert isinstance(mem, RewriteMemory)
        assert mem.language == "chinese"

    def test_create_rag(self):
        mem = create_memory(memory_type="rag", language="chinese")
        assert isinstance(mem, RAGMemory)

    def test_create_null(self):
        mem = create_memory(memory_type="null", language="chinese")
        assert isinstance(mem, NullMemory)

    def test_create_full_context(self):
        # Override tokenizer_model to avoid network download of o200k_base
        mem = create_memory(
            memory_type="full_context", language="chinese", tokenizer_model="cl100k_base"
        )
        assert isinstance(mem, FullContextMemory)

    def test_default_is_rewrite(self):
        mem = create_memory(language="chinese")
        assert isinstance(mem, RewriteMemory)


class TestCreateMemoryErrors:

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown memory type"):
            create_memory(memory_type="nonexistent")

    def test_invalid_class_path_raises(self):
        with pytest.raises((ImportError, AttributeError, ModuleNotFoundError)):
            create_memory(memory_class="nonexistent.module.FakeClass")


class TestCreateMemoryCustomClass:

    def test_custom_class_by_dotpath(self):
        mem = create_memory(
            memory_class="vita.memory.null_memory.NullMemory",
            language="english",
        )
        assert isinstance(mem, NullMemory)
        assert mem.language == "english"

    def test_custom_class_overrides_type(self):
        """When both memory_class and memory_type are given, memory_class wins."""
        with pytest.warns(UserWarning):
            mem = create_memory(
                memory_type="rewrite",
                memory_class="vita.memory.null_memory.NullMemory",
                language="chinese",
            )
        assert isinstance(mem, NullMemory)


class TestRegistry:

    def test_all_registry_entries_importable(self):
        for name, dotpath in MEMORY_REGISTRY.items():
            mem = create_memory(memory_type=name, language="chinese")
            assert isinstance(mem, BaseMemory), f"{name} did not produce a BaseMemory"

    @pytest.mark.parametrize("name", ["rewrite", "rag", "null", "full_context"])
    def test_registry_entry_exists(self, name):
        assert name in MEMORY_REGISTRY


class TestCreateMemoryKwargs:

    def test_top_k_passed_through(self):
        mem = create_memory(memory_type="rag", language="chinese", top_k=10)
        assert mem.top_k == 10

    def test_similarity_threshold_passed_through(self):
        mem = create_memory(memory_type="rag", language="chinese", similarity_threshold=0.5)
        assert mem.similarity_threshold == 0.5

    def test_full_context_max_tokens(self):
        mem = create_memory(memory_type="full_context", language="chinese", max_tokens=2048)
        assert mem.max_tokens == 2048
