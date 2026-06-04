"""Tests for read(query) across all memory backends."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestNullMemoryRead:
    def test_read_no_query(self):
        from vita.memory.null_memory import NullMemory
        mem = NullMemory(language="chinese")
        assert mem.read() == ""

    def test_read_with_query_still_empty(self):
        from vita.memory.null_memory import NullMemory
        mem = NullMemory(language="chinese")
        assert mem.read(query="order food") == ""

    def test_receives_top_k(self):
        from vita.memory.null_memory import NullMemory
        mem = NullMemory(language="chinese", top_k=10)
        assert mem.top_k == 10


class TestRewriteMemoryRead:
    def test_read_no_query_empty(self):
        from vita.memory.rewrite_memory import RewriteMemory
        mem = RewriteMemory(language="chinese")
        assert "No user preference" in mem.read()

    def test_read_with_query_returns_same_as_no_query(self):
        from vita.memory.rewrite_memory import RewriteMemory
        mem = RewriteMemory(language="chinese")
        mem._memory_text = "User likes spicy food"
        assert mem.read(query="food preference") == mem.read()

    def test_receives_top_k(self):
        from vita.memory.rewrite_memory import RewriteMemory
        mem = RewriteMemory(language="chinese", top_k=3)
        assert mem.top_k == 3


class TestRAGMemoryRead:
    def _make_rag_with_chunks(self):
        from vita.memory.rag_memory import RAGMemory
        mem = RAGMemory(language="chinese", top_k=2)
        mem._chunks = [
            {"text": "用户喜欢麻辣火锅", "keywords": ["麻辣", "火锅"], "timestamp": "2024-01-01", "type": "order"},
            {"text": "用户经常点奶茶", "keywords": ["奶茶"], "timestamp": "2024-01-02", "type": "order"},
            {"text": "用户住在朝阳区", "keywords": ["朝阳区", "住址"], "timestamp": "2024-01-03", "type": "browse"},
            {"text": "用户预订了酒店", "keywords": ["酒店", "预订"], "timestamp": "2024-01-04", "type": "order"},
        ]
        return mem

    def test_read_no_query_returns_full_summary(self):
        mem = self._make_rag_with_chunks()
        result = mem.read()
        assert "麻辣火锅" in result
        assert "奶茶" in result
        assert "酒店" in result

    def test_read_with_query_returns_relevant_only(self):
        mem = self._make_rag_with_chunks()
        result = mem.read(query="火锅 麻辣")
        assert "麻辣火锅" in result
        # top_k=2, so at most 2 results
        lines = [l for l in result.split("\n") if l.startswith("- ")]
        assert len(lines) <= 2

    def test_read_with_query_no_match_falls_back(self):
        mem = self._make_rag_with_chunks()
        result = mem.read(query="完全无关的查询xyz")
        # No matches -> falls back to full summary
        assert "麻辣火锅" in result

    def test_read_empty_chunks(self):
        from vita.memory.rag_memory import RAGMemory
        mem = RAGMemory(language="chinese")
        assert "No user preference" in mem.read(query="anything")

    def test_top_k_limits_results(self):
        mem = self._make_rag_with_chunks()
        mem.top_k = 1
        result = mem.read(query="火锅 奶茶")
        lines = [l for l in result.split("\n") if l.startswith("- ")]
        assert len(lines) == 1

    def test_score_chunks_shared_with_tool(self):
        mem = self._make_rag_with_chunks()
        scored = mem._score_chunks("火锅")
        assert len(scored) > 0
        assert scored[0][1]["text"] == "用户喜欢麻辣火锅"


