"""
RAGCacheMemory: RAG memory backed by a pre-computed embedding cache.

Behavioural contract mirrors RAGMemory:
  - update() appends chunks to an accumulating in-memory index
  - read(query=...) returns top-k chunks by cosine similarity,
    filtered by similarity_threshold

Difference: no HTTP calls. All chunk embeddings and the query embedding for
each subtask are loaded from a .npz produced by
scripts/precompute_rag_cache.py.

Strict mode: any miss — unknown (user_id, subtask_id) file, or read(query)
whose query text is not in the cached query index — raises
RAGCacheMissError. No silent fallback to live embedding, no keyword fallback.

Cache file layout (one per subtask):
  {cache_dir}/{user_id}__{subtask_id}.npz
  keys:
    chunk_embeddings : (N, D) float32
    chunk_texts      : (N,)   object   (str)
    chunk_keywords   : (N,)   object   (list[str])
    chunk_timestamps : (N,)   object   (str)
    chunk_types      : (N,)   object   (str)
    query_embedding  : (D,)   float32
    instruction      : scalar object   (str) — cache key for read(query=...)
    model            : scalar object   (str) — embedding model identifier
    chunk_size       : scalar int
    chunk_overlap    : scalar int
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from loguru import logger

from vita.environment.toolkit import ToolType, is_tool
from vita.memory.base import BaseMemory


_DEFAULT_CACHE_DIR = "data/vita/domains/personalization/rag_cache"


class RAGCacheMissError(RuntimeError):
    """Raised when a required embedding is not in the cache."""


class RAGCacheMemory(BaseMemory):
    """RAG memory backed by a pre-computed embedding cache on disk.

    Same retrieval semantics as RAGMemory (cosine top-k with threshold),
    but embeddings are loaded from .npz files produced offline. Never calls
    the embedding API.

    The orchestrator must call `set_current_location(user_id, subtask_id)`
    before `update()` so the memory knows which cache file to load.
    """

    def __init__(
        self,
        language: str = None,
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(language=language, **kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        resolved = cache_dir or os.environ.get("VITA_RAG_CACHE_DIR", _DEFAULT_CACHE_DIR)
        self._cache_dir = Path(resolved)

        self._chunks: List[dict] = []
        self._embeddings: List[np.ndarray] = []
        # instruction text -> query embedding
        self._query_cache: dict[str, np.ndarray] = {}

        self._current_user_id: Optional[str] = None
        self._current_subtask_id: Optional[str] = None

    # ── Orchestrator hook ────────────────────────────────────────────────────

    def set_current_location(self, user_id: str, subtask_id: str) -> None:
        """Called by PersonalizationOrchestrator before each subtask.

        Tells us which cache file to load on the next `update()` call.
        """
        self._current_user_id = str(user_id)
        self._current_subtask_id = str(subtask_id)

    # ── BaseMemory interface ─────────────────────────────────────────────────

    def update(
        self,
        new_interactions: list,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        **kwargs,
    ) -> str:
        """Append this subtask's pre-computed chunks/query to the in-memory index.

        `new_interactions` is accepted for API compatibility and to size-check
        against the cache file; the actual chunk texts and embeddings come
        from disk, not from re-deriving them here.
        """
        if not new_interactions:
            return self.read()

        if self._current_user_id is None or self._current_subtask_id is None:
            raise RAGCacheMissError(
                "RAGCacheMemory.update() called without set_current_location(); "
                "PersonalizationOrchestrator must set the (user_id, subtask_id) "
                "before each subtask."
            )

        path = self._cache_path(self._current_user_id, self._current_subtask_id)
        if not path.exists():
            raise RAGCacheMissError(
                f"No cache file for user_id={self._current_user_id!r} "
                f"subtask_id={self._current_subtask_id!r}: expected {path}"
            )

        with np.load(path, allow_pickle=True) as data:
            chunk_embeddings = np.asarray(data["chunk_embeddings"], dtype=np.float32)
            chunk_texts = data["chunk_texts"]
            chunk_keywords = data["chunk_keywords"]
            chunk_timestamps = data["chunk_timestamps"]
            chunk_types = data["chunk_types"]
            query_embedding = np.asarray(data["query_embedding"], dtype=np.float32)
            instruction = str(data["instruction"])

        n = len(chunk_texts)
        if chunk_embeddings.shape[0] != n:
            raise RAGCacheMissError(
                f"Malformed cache file {path}: embeddings shape "
                f"{chunk_embeddings.shape} inconsistent with {n} chunk texts"
            )

        for i in range(n):
            self._chunks.append(
                {
                    "text": str(chunk_texts[i]),
                    "keywords": list(chunk_keywords[i]) if chunk_keywords[i] is not None else [],
                    "timestamp": str(chunk_timestamps[i]),
                    "type": str(chunk_types[i]),
                }
            )
            self._embeddings.append(chunk_embeddings[i])

        self._query_cache[instruction] = query_embedding

        logger.debug(
            f"RAGCacheMemory: loaded {n} chunks for "
            f"({self._current_user_id}, {self._current_subtask_id}); "
            f"total chunks now={len(self._chunks)}"
        )

        return self._build_summary()

    def read(self, query: Optional[str] = None) -> str:
        """Return memory content. Miss → raise."""
        if not self._chunks:
            return "No user preference information available yet."

        if not query:
            return self._build_summary()

        if query not in self._query_cache:
            raise RAGCacheMissError(
                f"No cached query embedding for query: {query!r}. "
                f"The precompute script only embeds `subtask.instruction` strings; "
                f"this query was not one of them."
            )

        q_emb = self._query_cache[query]
        scored = self._score_chunks_vector(q_emb)
        if not scored:
            return self._build_summary()

        top = scored[: self.top_k]
        lines = [
            f"- [{c['timestamp']}] [{c['type']}] {c['text']}"
            for _, c in top
        ]
        return "User preference memory (retrieved for current task):\n" + "\n".join(lines)

    def reset(self) -> None:
        self._chunks.clear()
        self._embeddings.clear()
        self._query_cache.clear()
        self._current_user_id = None
        self._current_subtask_id = None

    # ── Tools exposed to the agent ───────────────────────────────────────────
    #
    # By design: only `read_preference_memory` is exposed. The original
    # `query_preference_memory(query)` tool is intentionally NOT defined here —
    # agent-generated query strings can't be pre-embedded, and strict mode
    # would force a RAGCacheMissError every time the agent tried to use it.

    @is_tool(ToolType.READ)
    def read_preference_memory(self) -> str:
        """读取用户偏好记忆，获取关于用户偏好的完整知识"""
        return self.read()

    # ── Internals ────────────────────────────────────────────────────────────

    def _cache_path(self, user_id: str, subtask_id: str) -> Path:
        return self._cache_dir / f"{user_id}__{subtask_id}.npz"

    def _score_chunks_vector(self, query_emb: np.ndarray) -> list[tuple]:
        q = np.asarray(query_emb, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            return []
        scored = []
        for chunk, emb in zip(self._chunks, self._embeddings):
            denom = q_norm * float(np.linalg.norm(emb))
            if denom == 0.0:
                continue
            sim = float(np.dot(q, emb) / denom)
            if sim >= self.similarity_threshold:
                scored.append((sim, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def _build_summary(self) -> str:
        if not self._chunks:
            return "No user preference information available."
        lines = [
            f"- [{c['timestamp']}] [{c['type']}] {c['text']}"
            for c in self._chunks
        ]
        return "User interaction history and preferences:\n" + "\n".join(lines)
