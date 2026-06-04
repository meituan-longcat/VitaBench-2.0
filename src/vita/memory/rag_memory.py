"""
RAGMemory: A vector-store based memory backend.

On update:
  1. Each new interaction is rendered to a text blob (_interaction_to_chunk).
  2. The blob is split into fixed-size token chunks (chunk_size / chunk_overlap)
     using the same tiktoken cl100k_base encoding as FullContextMemory.
  3. Each chunk is embedded via AsyncOpenAI.embeddings.create (bounded
     concurrency, with exponential back-off), using the base_url + api_key
     resolved from models.yaml or env-var overrides (see below).
  4. Embeddings are stored for cosine-similarity retrieval.
On query:
  Compute query embedding, rank chunks by cosine similarity, return top-k.
read():
  Returns a summary of top-k chunks retrieved for the current task instruction,
  or a full dump when no query is supplied.

Embedding configuration (all optional, env-var overrides available):
  VITA_EMBEDDING_MODEL   — model name (default: text-embedding-3-large)
  VITA_EMBEDDING_URL     — base_url for the AsyncOpenAI client; falls back to
                           models.yaml default.base_url
  VITA_EMBEDDING_KEY     — api_key for the AsyncOpenAI client; falls back to
                           models.yaml default.api_key

Tokenizer:
  tiktoken cl100k_base, loaded from TIKTOKEN_CACHE_DIR. When tiktoken is
  unavailable, falls back to a chars/token=2 heuristic (matches FullContextMemory).
"""

import asyncio
import json
import os
import threading
import weakref
from typing import List, Optional

import numpy as np
from loguru import logger
from openai import AsyncOpenAI

from vita.memory.base import BaseMemory
from vita.environment.toolkit import ToolType, is_tool


# ── Tokenizer (shared setup with FullContextMemory) ───────────────────────────

_DEFAULT_TIKTOKEN_CACHE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "tiktoken_cache")
)
if not os.environ.get("TIKTOKEN_CACHE_DIR") and os.path.isdir(_DEFAULT_TIKTOKEN_CACHE):
    os.environ["TIKTOKEN_CACHE_DIR"] = _DEFAULT_TIKTOKEN_CACHE

try:
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODING = None

_CHARS_PER_TOKEN = 2  # fallback ratio when tiktoken is unavailable


def _split_text_by_tokens(
    text: str, chunk_size: int, chunk_overlap: int = 0
) -> List[str]:
    """Split text into fixed-size token chunks with optional overlap.

    Uses tiktoken cl100k_base when available; otherwise splits by character
    count (chunk_size * _CHARS_PER_TOKEN) as a rough approximation.
    """
    if not text:
        return []
    if chunk_size <= 0:
        return [text]
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        chunk_overlap = 0

    stride = chunk_size - chunk_overlap

    if _ENCODING is not None:
        ids = _ENCODING.encode(text)
        if len(ids) <= chunk_size:
            return [text]
        pieces = []
        for start in range(0, len(ids), stride):
            window = ids[start : start + chunk_size]
            if not window:
                break
            pieces.append(_ENCODING.decode(window))
            if start + chunk_size >= len(ids):
                break
        return pieces

    # Fallback: character-based approximation
    char_size = chunk_size * _CHARS_PER_TOKEN
    char_stride = stride * _CHARS_PER_TOKEN
    if len(text) <= char_size:
        return [text]
    pieces = []
    for start in range(0, len(text), char_stride):
        window = text[start : start + char_size]
        if not window:
            break
        pieces.append(window)
        if start + char_size >= len(text):
            break
    return pieces


# ── Embedding endpoint resolution ─────────────────────────────────────────────

_EMBED_MODEL_DEFAULT = "text-embedding-3-large"
_MAX_CONCURRENCY_DEFAULT = int(os.environ.get("VITA_EMBEDDING_MAX_CONCURRENCY", "64"))

# Per-event-loop asyncio.Semaphore cache.
#
# Why per-loop instead of a single threading.BoundedSemaphore:
# the previous threading gate was acquired inside the asyncio loop via
# `loop.run_in_executor(None, gate.acquire)`. When INNER_CONCURRENCY threads
# each ran their own event loop and each gathered hundreds of embed coroutines
# (e.g. 4 trials × 76 historical_behavior chunks = 304 queued acquires), the
# default ThreadPoolExecutor (max_workers ≈ 32) filled up with blocked
# gate.acquire() calls and new acquires queued behind *the executor itself*,
# not the semaphore. Net effect: only 4 POSTs actually left the process at any
# moment, and the rest timed out at the HTTP client's 60s ceiling in waves.
#
# A plain asyncio.Semaphore used via `async with sem:` never enters the
# executor, so backlog is bounded by the loop itself and no threads are
# wasted. Cap is now per-event-loop; with INNER_CONCURRENCY N threads each
# running one loop, total in-flight embeds ≤ N × _MAX_CONCURRENCY_DEFAULT.
# Sizing VITA_EMBEDDING_MAX_CONCURRENCY modestly (e.g. 4) keeps total embeds
# ≈ INNER × 4, still well under the 1000 RPM embedding key.
_LOOP_EMBED_SEMS: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = (
    weakref.WeakKeyDictionary()
)
_LOOP_EMBED_SEMS_LOCK = threading.Lock()


def _get_loop_embed_sem() -> asyncio.Semaphore:
    loop = asyncio.get_event_loop()
    sem = _LOOP_EMBED_SEMS.get(loop)
    if sem is None:
        with _LOOP_EMBED_SEMS_LOCK:
            sem = _LOOP_EMBED_SEMS.get(loop)
            if sem is None:
                sem = asyncio.Semaphore(_MAX_CONCURRENCY_DEFAULT)
                _LOOP_EMBED_SEMS[loop] = sem
    return sem


def _embed_client() -> tuple[AsyncOpenAI, str]:
    """Return (AsyncOpenAI client, embedding model name).

    Resolution order for base_url: VITA_EMBEDDING_URL → models.yaml default.base_url.
    Resolution order for api_key:  VITA_EMBEDDING_KEY → models.yaml default.api_key.
    Resolution order for model:    VITA_EMBEDDING_MODEL → text-embedding-3-large.

    `max_retries=0` is mandatory: we keep our own retry/back-off loop in
    `_embed_one_with_backoff`, with logging the SDK doesn't expose. Letting
    the SDK's retry layer compose with ours would silently double the
    effective retry budget and obscure backoff timings.
    """
    from vita.config import models  # lazy import to avoid circular deps

    default_cfg = models.get("default", {})
    base_url = os.environ.get("VITA_EMBEDDING_URL") or default_cfg.get("base_url")
    api_key = os.environ.get("VITA_EMBEDDING_KEY") or default_cfg.get("api_key")
    model = os.environ.get("VITA_EMBEDDING_MODEL", _EMBED_MODEL_DEFAULT)

    if not base_url or not api_key:
        raise RuntimeError(
            "Embedding endpoint not configured. Set VITA_EMBEDDING_URL/VITA_EMBEDDING_KEY "
            "or provide default.base_url/default.api_key in models.yaml."
        )
    return AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0), model


# ── Async embedding with exponential back-off ─────────────────────────────────

async def _embed_one_with_backoff(
    client: AsyncOpenAI,
    model: str,
    text: str,
    max_retries: int = 6,
) -> List[float]:
    """Embed a single text with exponential back-off.

    Each call acquires the per-event-loop embedding semaphore before firing,
    so total in-flight requests on a given event loop never exceed
    VITA_EMBEDDING_MAX_CONCURRENCY. Spec §4.5: only the transport changes
    (raw HTTP POST → AsyncOpenAI.embeddings.create); the semaphore +
    retry/back-off schedule are preserved verbatim from the pre-rewrite
    implementation (delay starts at 1.0s, doubles to a 60.0s cap).
    """
    delay = 1.0
    sem = _get_loop_embed_sem()
    for attempt in range(max_retries + 1):
        caught: Optional[Exception] = None
        async with sem:
            try:
                resp = await client.embeddings.create(
                    model=model, input=text.replace("\n", " ")
                )
                return resp.data[0].embedding
            except Exception as e:
                caught = e
        if caught is not None:
            if attempt == max_retries:
                raise caught
            err_repr = repr(caught) if not str(caught) else str(caught)
            logger.warning(
                f"Embedding attempt {attempt + 1} failed ({err_repr}); "
                f"retrying in {delay:.1f}s"
            )
            # Sleep outside the semaphore so a backed-off retry doesn't hold
            # a slot away from other coroutines.
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60.0)
    raise RuntimeError("Embedding failed after all retries")  # unreachable


async def _embed_batch_async(
    texts: List[str], max_concurrency: int = _MAX_CONCURRENCY_DEFAULT
) -> List[List[float]]:
    """Embed multiple texts concurrently via the AsyncOpenAI client.

    NOTE: `max_concurrency` is a no-op for back-compat — the per-event-loop
    semaphore inside `_embed_one_with_backoff` (sized by
    VITA_EMBEDDING_MAX_CONCURRENCY) is what actually caps in-flight requests.
    Kept as a parameter for API compatibility with existing callers.
    """
    if not texts:
        return []
    client, model = _embed_client()
    return await asyncio.gather(
        *[_embed_one_with_backoff(client, model, t) for t in texts]
    )


def embed_texts(texts: List[str], max_concurrency: int = _MAX_CONCURRENCY_DEFAULT) -> List[List[float]]:
    """Synchronous entry-point: embed a list of texts concurrently."""
    if not texts:
        return []
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Running inside an existing event-loop (e.g. Jupyter / nest_asyncio)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(asyncio.run, _embed_batch_async(texts, max_concurrency))
                return fut.result()
        else:
            return loop.run_until_complete(_embed_batch_async(texts, max_concurrency))
    except RuntimeError:
        return asyncio.run(_embed_batch_async(texts, max_concurrency))


# ── Cosine similarity ─────────────────────────────────────────────────────────

def _cosine(a: List[float], b: List[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


# ── RAGMemory ─────────────────────────────────────────────────────────────────

class RAGMemory(BaseMemory):
    """RAG-based memory using embedding-based vector retrieval.

    Chunks are embedded asynchronously (up to 64 concurrent requests, with
    exponential back-off) on update(), via AsyncOpenAI configured with
    base_url + api_key from models.yaml default (or VITA_EMBEDDING_URL /
    VITA_EMBEDDING_KEY env overrides).  Retrieval uses cosine similarity
    against the stored embedding index — no external vector DB required.

    Falls back gracefully to keyword matching if the embedding call fails.
    """

    def __init__(
        self,
        language: str = None,
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        **kwargs,
    ):
        super().__init__(language=language, **kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._chunks: List[dict] = []
        # Parallel list of embeddings; None means not yet computed / unavailable
        self._embeddings: List[Optional[List[float]]] = []
        self._summary: str = ""

    # ── Internal retrieval ────────────────────────────────────────────────────

    def _get_query_embedding(self, query: str) -> Optional[List[float]]:
        """Embed the query string; return None on failure."""
        try:
            return embed_texts([query], max_concurrency=1)[0]
        except Exception as e:
            logger.warning(f"RAGMemory: query embedding failed ({e}); falling back to keyword search")
            return None

    def _score_chunks_vector(self, query_emb: List[float]) -> List[tuple]:
        """Rank chunks by cosine similarity to the query embedding."""
        scored = []
        for i, chunk in enumerate(self._chunks):
            emb = self._embeddings[i]
            if emb is None:
                continue
            sim = _cosine(query_emb, emb)
            if sim >= self.similarity_threshold:
                scored.append((sim, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def _score_chunks_keyword(self, query: str) -> List[tuple]:
        """Fallback: keyword overlap scoring (original behaviour)."""
        query_words = set(query.lower().split())
        scored = []
        for chunk in self._chunks:
            kw = set(k.lower() for k in chunk["keywords"])
            tw = set(chunk["text"].lower().split())
            score = len(query_words & kw) + len(query_words & tw) * 0.5
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def _score_chunks(self, query: str) -> List[tuple]:
        """Try vector search; fall back to keyword if embeddings unavailable."""
        query_emb = self._get_query_embedding(query)
        if query_emb is not None:
            results = self._score_chunks_vector(query_emb)
            if results:
                return results
        return self._score_chunks_keyword(query)

    # ── BaseMemory interface ──────────────────────────────────────────────────

    def read(self, query: str = None) -> str:
        """Return memory content. When query is provided, retrieve relevant chunks."""
        if not self._chunks:
            return "No user preference information available yet."

        if query:
            scored = self._score_chunks(query)
            if scored:
                top = scored[: self.top_k]
                lines = [
                    f"- [{c['timestamp']}] [{c['type']}] {c['text']}"
                    for _, c in top
                ]
                return "User preference memory (retrieved for current task):\n" + "\n".join(lines)

        if self._summary:
            return self._summary

        return self._build_summary()

    def update(
        self,
        new_interactions: list,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        **kwargs,
    ) -> str:
        """Store new interactions and compute their embeddings.

        Embeddings for all new chunks are requested in a single batched
        async call so concurrent requests are bounded and retried with
        exponential back-off.
        """
        if not new_interactions:
            return self.read()

        # 1. Render each interaction to a base chunk (carries keywords/timestamp/type).
        base_chunks = [self._interaction_to_chunk(i) for i in new_interactions]

        # 2. Split each base chunk's text into fixed-size token sub-chunks.
        sub_chunks: List[dict] = []
        for base in base_chunks:
            pieces = _split_text_by_tokens(
                base["text"], self.chunk_size, self.chunk_overlap
            )
            if not pieces:
                continue
            for piece in pieces:
                sub_chunks.append(
                    {
                        "text": piece,
                        "keywords": base["keywords"],
                        "timestamp": base["timestamp"],
                        "type": base["type"],
                    }
                )

        if not sub_chunks:
            if llm is not None:
                self._summary = self._generate_summary(llm, llm_args)
            else:
                self._summary = self._build_summary()
            return self._summary

        # 3. Batch-embed all sub-chunks.
        texts = [c["text"] for c in sub_chunks]
        try:
            new_embeddings = embed_texts(texts)
        except Exception as e:
            logger.warning(f"RAGMemory: batch embedding failed ({e}); storing chunks without embeddings")
            new_embeddings = [None] * len(texts)

        for chunk, emb in zip(sub_chunks, new_embeddings):
            self._chunks.append(chunk)
            self._embeddings.append(emb)

        if llm is not None:
            self._summary = self._generate_summary(llm, llm_args)
        else:
            self._summary = self._build_summary()

        return self._summary

    # ── Tools: auto-discovered by framework via @is_tool ─────────────────────

    @is_tool(ToolType.READ)
    def read_preference_memory(self) -> str:
        """读取用户偏好记忆，获取关于用户偏好的完整知识"""
        return self.read()

    @is_tool(ToolType.READ)
    def query_preference_memory(self, query: str) -> str:
        """根据具体问题查询用户偏好记忆"""
        if not self._chunks:
            return "No user preference information available."

        scored = self._score_chunks(query)
        if not scored:
            return self._build_summary()

        lines = [
            f"- [{c['timestamp']}] [{c['type']}] {c['text']}"
            for _, c in scored[: self.top_k]
        ]
        return "\n".join(lines)

    def reset(self):
        """Reset memory to empty state."""
        self._chunks = []
        self._embeddings = []
        self._summary = ""

    # ── Chunk conversion ─────────────────────────────────────────────────────

    @staticmethod
    def _interaction_to_chunk(interaction) -> dict:
        """Convert an interaction to a storable chunk.

        Supports both Interaction objects ({type, timestamp, content}) and
        init_gen format dicts ({date, behavior, dialogue}).
        """
        if isinstance(interaction, dict):
            if "date" in interaction and ("behavior" in interaction or "dialogue" in interaction):
                text_parts = []
                keywords = []
                for beh in interaction.get("behavior", []):
                    if isinstance(beh, dict):
                        text_parts.append(json.dumps(beh, ensure_ascii=False))
                        content = beh.get("content", {})
                        if isinstance(content, dict):
                            keywords.extend(str(v) for v in content.values() if isinstance(v, str))
                dialogue = interaction.get("dialogue", [])
                if dialogue:
                    for msg in dialogue:
                        if isinstance(msg, dict) and "content" in msg:
                            keywords.extend(str(msg["content"]).split()[:5])
                    text_parts.append(json.dumps(dialogue, ensure_ascii=False))
                return {
                    "text": "\n".join(text_parts),
                    "keywords": keywords,
                    "timestamp": interaction.get("date", ""),
                    "type": "daily_record",
                }
            content = interaction.get("content", interaction)
            content_str = (
                json.dumps(content, ensure_ascii=False)
                if isinstance(content, (dict, list))
                else str(content)
            )
            keywords = []
            if isinstance(content, dict):
                keywords.extend(str(v) for v in content.values() if isinstance(v, str))
                keywords.extend(content.keys())
            return {
                "text": content_str,
                "keywords": keywords,
                "timestamp": interaction.get("timestamp", ""),
                "type": interaction.get("type", "unknown"),
            }

        # Pydantic Interaction object
        content_str = (
            json.dumps(interaction.content, ensure_ascii=False)
            if isinstance(interaction.content, (dict, list))
            else str(interaction.content)
        )
        keywords = []
        if isinstance(interaction.content, dict):
            keywords.extend(str(v) for v in interaction.content.values() if isinstance(v, str))
            keywords.extend(interaction.content.keys())
        elif isinstance(interaction.content, str):
            keywords = interaction.content.split()[:10]

        return {
            "text": content_str,
            "keywords": keywords,
            "timestamp": interaction.timestamp,
            "type": interaction.type,
        }

    # ── Summary helpers ───────────────────────────────────────────────────────

    def _build_summary(self) -> str:
        """Build a text summary from all stored chunks."""
        if not self._chunks:
            return "No user preference information available."
        lines = [
            f"- [{c['timestamp']}] [{c['type']}] {c['text']}"
            for c in self._chunks
        ]
        return "User interaction history and preferences:\n" + "\n".join(lines)

    def _generate_summary(self, llm: str, llm_args: Optional[dict] = None) -> str:
        """Use LLM to generate a concise preference summary from all chunks."""
        from vita.utils.llm_utils import generate
        from vita.data_model.message import SystemMessage, UserMessage

        all_chunks_text = "\n".join(
            f"- [{c['timestamp']}] ({c['type']}): {c['text']}"
            for c in self._chunks
        )

        messages = [
            SystemMessage(
                role="system",
                content="You are a preference memory manager. Summarize the user's preferences based on their interaction history.",
            ),
            UserMessage(
                role="user",
                content=(
                    f"Based on the following user interactions, generate a concise summary of the user's preferences:\n\n"
                    f"{all_chunks_text}\n\n"
                    f"Provide a structured summary of preferences including food preferences, "
                    f"spending habits, location preferences, timing preferences, etc."
                ),
            ),
        ]

        if llm_args is None:
            llm_args = {}

        try:
            response = generate(
                model=llm,
                messages=messages,
                tools=None,
                **llm_args,
            )
            if response is not None and response.content:
                return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating RAG summary: {e}")

        return self._build_summary()
