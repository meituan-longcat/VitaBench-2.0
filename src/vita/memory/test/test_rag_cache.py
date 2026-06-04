"""Tests for RAGCacheMemory.

Covers:
- update() loads a cached subtask and appends chunks
- read(query=cached) returns top-k by cosine similarity
- read(query=uncached) raises RAGCacheMissError
- update() against a missing cache file raises
- reset() clears all state
- MEMORY_REGISTRY and YAML config expose rag_cache
- Only `read_preference_memory` is exposed as a tool (no query_preference_memory)
"""

import numpy as np
import pytest

from vita.memory.rag_cache import RAGCacheMemory, RAGCacheMissError


EMBED_DIM = 8


def _write_fake_cache(
    cache_dir,
    user_id: str,
    subtask_id: str,
    chunk_texts: list[str],
    chunk_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    instruction: str,
    model: str = "text-embedding-3-large",
    chunk_size: int = 512,
    chunk_overlap: int = 0,
):
    path = cache_dir / f"{user_id}__{subtask_id}.npz"
    n = len(chunk_texts)
    np.savez_compressed(
        path,
        chunk_embeddings=chunk_embeddings.astype(np.float32),
        chunk_texts=np.asarray(chunk_texts, dtype=object),
        chunk_keywords=np.asarray([[] for _ in range(n)], dtype=object),
        chunk_timestamps=np.asarray([f"2023-03-{i+1:02d}" for i in range(n)], dtype=object),
        chunk_types=np.asarray(["daily_record"] * n, dtype=object),
        query_embedding=query_embedding.astype(np.float32),
        instruction=instruction,
        model=model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return path


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    cache_dir = tmp_path / "rag_cache"
    cache_dir.mkdir()
    monkeypatch.setenv("VITA_RAG_CACHE_DIR", str(cache_dir))
    return cache_dir


class TestRAGCacheBasic:
    def test_initial_read_empty(self, tmp_cache):
        mem = RAGCacheMemory(language="chinese")
        assert "No user preference" in mem.read()

    def test_update_without_set_current_location_raises(self, tmp_cache):
        mem = RAGCacheMemory(language="chinese")
        with pytest.raises(RAGCacheMissError, match="set_current_location"):
            mem.update([{"date": "2023-03-01", "behavior": [], "dialogue": []}])

    def test_update_missing_file_raises(self, tmp_cache):
        mem = RAGCacheMemory(language="chinese")
        mem.set_current_location("ghost_user", "ghost_subtask")
        with pytest.raises(RAGCacheMissError, match="No cache file"):
            mem.update([{"date": "2023-03-01", "behavior": [], "dialogue": []}])


class TestRAGCacheRetrieval:
    def test_read_with_cached_query_returns_topk(self, tmp_cache):
        # Build 3 chunks whose embeddings have known cosine similarity to the query.
        query_emb = _unit(np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
        chunk_embs = np.stack(
            [
                _unit(np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)),  # sim 1.0
                _unit(np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)),  # sim 0.0 (filtered)
                _unit(np.array([0.8, 0.2, 0, 0, 0, 0, 0, 0], dtype=np.float32)),  # sim ~0.97
            ]
        )
        _write_fake_cache(
            tmp_cache,
            user_id="u1",
            subtask_id="s1",
            chunk_texts=["target_chunk", "irrelevant", "near_match"],
            chunk_embeddings=chunk_embs,
            query_embedding=query_emb,
            instruction="ask about target",
        )
        mem = RAGCacheMemory(language="chinese", top_k=2, similarity_threshold=0.3)
        mem.set_current_location("u1", "s1")
        mem.update([{"date": "2023-03-01", "behavior": [], "dialogue": []}])  # trigger load

        result = mem.read(query="ask about target")
        assert "retrieved" in result.lower()
        assert "target_chunk" in result
        assert "near_match" in result
        assert "irrelevant" not in result  # filtered by threshold

    def test_read_with_uncached_query_raises(self, tmp_cache):
        query_emb = _unit(np.ones(EMBED_DIM, dtype=np.float32))
        chunk_embs = _unit(np.ones((1, EMBED_DIM), dtype=np.float32))
        _write_fake_cache(
            tmp_cache, "u1", "s1",
            chunk_texts=["anything"],
            chunk_embeddings=chunk_embs,
            query_embedding=query_emb,
            instruction="cached question",
        )
        mem = RAGCacheMemory(language="chinese")
        mem.set_current_location("u1", "s1")
        mem.update([{"date": "2023-03-01", "behavior": [], "dialogue": []}])

        with pytest.raises(RAGCacheMissError, match="No cached query embedding"):
            mem.read(query="never-asked question")

    def test_read_without_query_returns_summary(self, tmp_cache):
        query_emb = _unit(np.ones(EMBED_DIM, dtype=np.float32))
        chunk_embs = _unit(np.ones((2, EMBED_DIM), dtype=np.float32))
        _write_fake_cache(
            tmp_cache, "u1", "s1",
            chunk_texts=["chunk_a", "chunk_b"],
            chunk_embeddings=chunk_embs,
            query_embedding=query_emb,
            instruction="instr",
        )
        mem = RAGCacheMemory(language="chinese")
        mem.set_current_location("u1", "s1")
        mem.update([{"date": "2023-03-01", "behavior": [], "dialogue": []}])

        result = mem.read()
        assert "chunk_a" in result and "chunk_b" in result


class TestRAGCacheMultiSubtask:
    def test_accumulates_across_subtasks(self, tmp_cache):
        q1 = _unit(np.array([1, 0] + [0] * (EMBED_DIM - 2), dtype=np.float32))
        q2 = _unit(np.array([0, 1] + [0] * (EMBED_DIM - 2), dtype=np.float32))
        e1 = _unit(np.ones((1, EMBED_DIM), dtype=np.float32))
        e2 = _unit(np.ones((1, EMBED_DIM), dtype=np.float32))
        _write_fake_cache(tmp_cache, "u1", "s1", ["c1"], e1, q1, "instr1")
        _write_fake_cache(tmp_cache, "u1", "s2", ["c2"], e2, q2, "instr2")

        mem = RAGCacheMemory(language="chinese", similarity_threshold=0.0)
        mem.set_current_location("u1", "s1")
        mem.update([{"date": "2023-03-01", "behavior": [], "dialogue": []}])
        mem.set_current_location("u1", "s2")
        mem.update([{"date": "2023-03-02", "behavior": [], "dialogue": []}])

        assert len(mem._chunks) == 2
        # Both cached queries should work after accumulation
        assert "c1" in mem.read(query="instr1") or "c2" in mem.read(query="instr1")
        assert "c2" in mem.read(query="instr2") or "c1" in mem.read(query="instr2")


class TestRAGCacheReset:
    def test_reset_clears_state(self, tmp_cache):
        q = _unit(np.ones(EMBED_DIM, dtype=np.float32))
        e = _unit(np.ones((1, EMBED_DIM), dtype=np.float32))
        _write_fake_cache(tmp_cache, "u1", "s1", ["c1"], e, q, "instr")

        mem = RAGCacheMemory(language="chinese")
        mem.set_current_location("u1", "s1")
        mem.update([{"date": "2023-03-01", "behavior": [], "dialogue": []}])
        assert len(mem._chunks) == 1

        mem.reset()
        assert mem._chunks == []
        assert mem._embeddings == []
        assert mem._query_cache == {}
        assert mem._current_user_id is None


class TestRAGCacheTools:
    def test_only_read_preference_memory_exposed(self, tmp_cache):
        mem = RAGCacheMemory(language="chinese")
        tool_names = set(mem.tools.keys())
        assert "read_preference_memory" in tool_names
        # Deliberately absent — agent-generated queries can't be pre-embedded.
        assert "query_preference_memory" not in tool_names

    def test_read_preference_memory_matches_read(self, tmp_cache):
        q = _unit(np.ones(EMBED_DIM, dtype=np.float32))
        e = _unit(np.ones((1, EMBED_DIM), dtype=np.float32))
        _write_fake_cache(tmp_cache, "u1", "s1", ["c1"], e, q, "instr")
        mem = RAGCacheMemory(language="chinese")
        mem.set_current_location("u1", "s1")
        mem.update([{"date": "2023-03-01", "behavior": [], "dialogue": []}])
        assert mem.read_preference_memory() == mem.read()


class TestRAGCacheRegistration:
    def test_registered_in_factory(self):
        from vita.memory import MEMORY_REGISTRY, create_memory
        assert "rag_cache" in MEMORY_REGISTRY
        # Factory can instantiate it (yaml config + language propagate)
        mem = create_memory(memory_type="rag_cache", language="chinese")
        assert isinstance(mem, RAGCacheMemory)
        assert mem.chunk_size == 512
        assert mem.chunk_overlap == 0
        assert mem.top_k == 8
        assert mem.similarity_threshold == 0.3
