"""
Tests for FullContextMemory.

Covers:
- Append-only behavior (no LLM)
- Token-based truncation (keeps tail)
- Multi-turn accumulation
"""

import pytest

from vita.memory.full_context import FullContextMemory


class TestFullContextMemoryBasic:

    def test_initial_read_empty(self):
        mem = FullContextMemory(language="chinese")
        assert mem.read() == ""

    def test_reset_clears(self, delivery_interactions_initgen):
        mem = FullContextMemory(language="chinese")
        mem.update(delivery_interactions_initgen)
        assert mem.read() != ""
        mem.reset()
        assert mem.read() == ""


class TestFullContextMemoryUpdate:

    def test_update_appends_formatted(self, delivery_interactions_initgen):
        mem = FullContextMemory(language="chinese")
        result = mem.update(delivery_interactions_initgen)
        assert "红烧肉盖饭" in result
        assert "川味盖饭王" in result
        assert "回锅肉" in result

    def test_update_empty_noop(self):
        mem = FullContextMemory(language="chinese")
        mem._memory_text = "existing"
        result = mem.update([])
        assert result == "existing"

    def test_update_accumulates(self, delivery_interactions_initgen, ota_interactions_initgen):
        mem = FullContextMemory(language="chinese")
        mem.update(delivery_interactions_initgen)
        mem.update(ota_interactions_initgen)
        result = mem.read()
        # Both sets of interactions present
        assert "红烧肉" in result or "盖饭" in result
        assert "青年旅舍" in result or "梦之旅" in result

    def test_all_domain_interactions(
        self,
        delivery_interactions_initgen,
        instore_interactions_initgen,
        ota_interactions_initgen,
    ):
        mem = FullContextMemory(language="chinese", max_tokens=8192)
        mem.update(delivery_interactions_initgen)
        mem.update(instore_interactions_initgen)
        mem.update(ota_interactions_initgen)
        result = mem.read()
        assert len(result) > 0


class TestFullContextMemoryTruncation:

    def test_small_max_tokens_truncates(self, delivery_interactions_initgen, ota_interactions_initgen):
        """With very small max_tokens, only recent content is kept."""
        mem = FullContextMemory(language="chinese", max_tokens=50)
        mem.update(delivery_interactions_initgen)
        mem.update(ota_interactions_initgen)
        result = mem.read()
        # The internal text is long, but read() truncates to ~50 tokens
        # The truncated text should be the tail (OTA data, added last)
        full_text = mem._memory_text
        assert len(result) < len(full_text)

    def test_large_max_tokens_no_truncation(self, delivery_interactions_initgen):
        mem = FullContextMemory(language="chinese", max_tokens=100000)
        mem.update(delivery_interactions_initgen)
        result = mem.read()
        assert result == mem._memory_text

    def test_truncation_keeps_tail(self):
        """Truncation should keep the most recent (tail) content."""
        mem = FullContextMemory(language="chinese", max_tokens=20)
        # First update: old data
        mem.update([
            {"type": "order", "timestamp": "2023-01-01", "content": "AAAA_OLD_DATA_AAAA"}
        ])
        # Second update: new data
        mem.update([
            {"type": "order", "timestamp": "2023-12-31", "content": "BBBB_NEW_DATA_BBBB"}
        ])
        result = mem.read()
        # New data (tail) should be preserved, old data (head) may be truncated
        assert "BBBB_NEW_DATA_BBBB" in result


class TestFullContextMemoryMultiTurn:

    def test_three_turn_evolution(self, multi_turn_interactions):
        mem = FullContextMemory(language="chinese", max_tokens=4096)
        for turn in multi_turn_interactions:
            mem.update(turn)

        result = mem.read()
        # All three turns should be present (within token budget)
        assert "川味盖饭王" in result or "红烧肉" in result
        assert "回锅肉" in result or "正宗" in result
        assert "青旅" in result or "梦之旅" in result


class TestFullContextMemoryNoTools:
    """FullContextMemory should not expose any @is_tool methods."""

    def test_no_tools(self):
        mem = FullContextMemory(language="chinese")
        assert len(mem.tools) == 0
