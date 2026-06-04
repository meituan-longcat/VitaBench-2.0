"""
Tests for NullMemory.

NullMemory is the no-op baseline: read/update return empty, reset is no-op.
"""

import pytest

from vita.memory.null_memory import NullMemory


class TestNullMemory:

    def test_read_returns_empty(self):
        mem = NullMemory(language="chinese")
        assert mem.read() == ""

    def test_read_with_query_returns_empty(self):
        mem = NullMemory(language="chinese")
        assert mem.read(query="红烧肉盖饭") == ""

    def test_update_returns_empty(self, delivery_interactions_initgen):
        mem = NullMemory(language="chinese")
        result = mem.update(delivery_interactions_initgen)
        assert result == ""

    def test_update_does_not_store(self, delivery_interactions_initgen):
        mem = NullMemory(language="chinese")
        mem.update(delivery_interactions_initgen)
        assert mem.read() == ""

    def test_reset_is_noop(self):
        mem = NullMemory(language="chinese")
        mem.reset()  # should not raise
        assert mem.read() == ""

    def test_no_tools_exposed(self):
        mem = NullMemory(language="chinese")
        assert len(mem.tools) == 0

    def test_multiple_updates_still_empty(
        self, delivery_interactions_initgen, ota_interactions_initgen
    ):
        mem = NullMemory(language="chinese")
        mem.update(delivery_interactions_initgen)
        mem.update(ota_interactions_initgen)
        assert mem.read() == ""
