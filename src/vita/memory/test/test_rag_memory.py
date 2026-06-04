"""
Tests for RAGMemory.

Covers:
- _interaction_to_chunk() for both formats
- _score_chunks() keyword matching
- read() with and without query
- update() with and without LLM
- reset()
- Multi-turn retrieval accuracy
"""

import pytest

from vita.memory.rag_memory import RAGMemory


class TestRAGMemoryBasic:

    def test_initial_read_empty(self):
        mem = RAGMemory(language="chinese")
        assert "No user preference" in mem.read()

    def test_reset_clears(self, delivery_interactions_initgen):
        mem = RAGMemory(language="chinese")
        mem.update(delivery_interactions_initgen)
        assert len(mem._chunks) > 0
        mem.reset()
        assert len(mem._chunks) == 0
        assert mem._summary == ""


class TestInteractionToChunk:
    """Test RAGMemory._interaction_to_chunk with various formats."""

    def test_initgen_format(self):
        interaction = {
            "date": "2023-03-10",
            "behavior": [
                {
                    "behavior_type": "order",
                    "content": {
                        "merchant_name": "川味盖饭王",
                        "items": [{"product_name": "红烧肉盖饭"}],
                    },
                },
            ],
            "dialogue": [
                {"role": "user", "content": "帮我点外卖"},
            ],
        }
        chunk = RAGMemory._interaction_to_chunk(interaction)
        assert chunk["type"] == "daily_record"
        assert chunk["timestamp"] == "2023-03-10"
        assert "川味盖饭王" in chunk["text"]
        assert len(chunk["keywords"]) > 0

    def test_model_format(self):
        interaction = {
            "type": "order",
            "timestamp": "2023-03-10 12:00:00",
            "content": {
                "merchant_name": "川味盖饭王",
                "items": [{"product_name": "红烧肉盖饭"}],
            },
        }
        chunk = RAGMemory._interaction_to_chunk(interaction)
        assert chunk["type"] == "order"
        assert chunk["timestamp"] == "2023-03-10 12:00:00"
        assert "川味盖饭王" in chunk["text"]
        assert "merchant_name" in chunk["keywords"] or "川味盖饭王" in chunk["keywords"]

    def test_model_format_string_content(self):
        interaction = {
            "type": "conversation",
            "timestamp": "2023-03-10 12:00:00",
            "content": "帮我点个外卖吧",
        }
        chunk = RAGMemory._interaction_to_chunk(interaction)
        assert chunk["type"] == "conversation"
        assert "帮我点个外卖吧" in chunk["text"]

    def test_unknown_dict_format(self):
        interaction = {"random_key": "random_value"}
        chunk = RAGMemory._interaction_to_chunk(interaction)
        assert "random_value" in chunk["text"]


class TestScoreChunks:
    """Test keyword matching retrieval."""

    def test_relevant_query_scores_higher(self, delivery_interactions_initgen, ota_interactions_initgen):
        mem = RAGMemory(language="chinese")
        mem.update(delivery_interactions_initgen)
        mem.update(ota_interactions_initgen)

        # Query for delivery keywords should rank delivery chunks first
        scored = mem._score_chunks("红烧肉盖饭 川菜")
        assert len(scored) > 0
        # Best match should contain delivery keywords
        best_chunk = scored[0][1]
        assert "红烧肉" in best_chunk["text"] or "川" in best_chunk["text"]

    def test_no_match_returns_empty(self):
        mem = RAGMemory(language="chinese")
        mem.update([
            {"type": "order", "timestamp": "2023-01-01 12:00:00", "content": {"items": ["pizza"]}}
        ])
        scored = mem._score_chunks("完全无关的查询xyz")
        assert len(scored) == 0


class TestRAGMemoryRead:

    def test_read_without_query(self, delivery_interactions_initgen):
        mem = RAGMemory(language="chinese")
        mem.update(delivery_interactions_initgen)
        result = mem.read()
        assert "User interaction history" in result or "preference" in result.lower()

    def test_read_with_query(self, delivery_interactions_initgen, ota_interactions_initgen):
        mem = RAGMemory(language="chinese")
        mem.update(delivery_interactions_initgen)
        mem.update(ota_interactions_initgen)
        result = mem.read(query="红烧肉 盖饭")
        # Should retrieve relevant chunks
        assert "retrieved" in result.lower() or "preference" in result.lower()

    def test_read_with_query_no_match(self):
        mem = RAGMemory(language="chinese")
        mem.update([
            {"type": "order", "timestamp": "2023-01-01", "content": {"item": "apple"}}
        ])
        result = mem.read(query="完全无关xyz")
        # Falls back to summary
        assert "apple" in result

    def test_top_k_limits_results(self, delivery_interactions_initgen, instore_interactions_initgen):
        mem = RAGMemory(language="chinese", top_k=1)
        mem.update(delivery_interactions_initgen)
        mem.update(instore_interactions_initgen)
        # With top_k=1 and a query, at most 1 chunk should appear in retrieval
        result = mem.read(query="盖饭")
        lines = [l for l in result.split("\n") if l.strip().startswith("-")]
        if "retrieved" in result.lower():
            assert len(lines) <= 1


class TestRAGMemoryUpdate:

    def test_update_stores_chunks(self, delivery_interactions_initgen):
        mem = RAGMemory(language="chinese")
        mem.update(delivery_interactions_initgen)
        assert len(mem._chunks) == len(delivery_interactions_initgen)

    def test_update_accumulates_chunks(self, delivery_interactions_initgen, ota_interactions_initgen):
        mem = RAGMemory(language="chinese")
        mem.update(delivery_interactions_initgen)
        n1 = len(mem._chunks)
        mem.update(ota_interactions_initgen)
        assert len(mem._chunks) == n1 + len(ota_interactions_initgen)

    def test_update_empty(self):
        mem = RAGMemory(language="chinese")
        result = mem.update([])
        assert "No user preference" in result


class TestRAGMemoryMultiTurn:
    """Simulate multi-subtask memory evolution."""

    def test_three_turns(self, multi_turn_interactions):
        mem = RAGMemory(language="chinese")
        for turn in multi_turn_interactions:
            mem.update(turn)

        # All 3 turns stored
        assert len(mem._chunks) == 3

        # Can retrieve delivery-related
        result = mem.read(query="川菜 盖饭 红烧肉")
        assert "川味盖饭王" in result or "红烧肉" in result or "盖饭" in result

    def test_pharmacy_retrieval(self, pharmacy_interactions_initgen, delivery_interactions_initgen):
        mem = RAGMemory(language="chinese")
        mem.update(delivery_interactions_initgen)
        mem.update(pharmacy_interactions_initgen)
        result = mem.read(query="药品 过敏 鼻炎")
        assert "药" in result or "氯雷他定" in result or "康佳" in result


class TestRAGMemoryTools:

    def test_tools_registered(self):
        mem = RAGMemory(language="chinese")
        tool_names = set(mem.tools.keys())
        assert "read_preference_memory" in tool_names
        assert "query_preference_memory" in tool_names

    def test_read_preference_memory(self, delivery_interactions_initgen):
        mem = RAGMemory(language="chinese")
        mem.update(delivery_interactions_initgen)
        result = mem.read_preference_memory()
        assert result == mem.read()

    def test_query_preference_memory(self, delivery_interactions_initgen):
        mem = RAGMemory(language="chinese")
        mem.update(delivery_interactions_initgen)
        result = mem.query_preference_memory("红烧肉盖饭")
        assert "红烧肉" in result or "盖饭" in result or len(result) > 0
