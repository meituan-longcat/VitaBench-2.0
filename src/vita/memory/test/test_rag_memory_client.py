"""Tests that the RAG embedding transport is wired through AsyncOpenAI.

After Chunk 6 of the OSS-prep plan, embeddings flow through
`openai.AsyncOpenAI.embeddings.create(...)` — not raw aiohttp POSTs to a
hardcoded gateway. Spec §4.5 mandates that only the transport changes; the
per-event-loop semaphore, the WeakKeyDictionary, and the retry/backoff loop
are preserved (covered by the broader test_rag_memory.py suite).
"""

import asyncio
from unittest.mock import patch, AsyncMock


def test_embed_client_uses_env_overrides(monkeypatch):
    """`_embed_client` must prefer VITA_EMBEDDING_URL / _KEY / _MODEL over models.yaml."""
    monkeypatch.setenv("VITA_EMBEDDING_URL", "https://override.example.com/v1")
    monkeypatch.setenv("VITA_EMBEDDING_KEY", "sk-override")
    monkeypatch.setenv("VITA_EMBEDDING_MODEL", "text-embedding-test")

    from vita.memory import rag_memory
    client, model = rag_memory._embed_client()

    assert client.base_url == "https://override.example.com/v1" or \
        str(client.base_url).startswith("https://override.example.com/v1")
    assert model == "text-embedding-test"


def test_embed_one_calls_async_openai(monkeypatch):
    """`_embed_one_with_backoff` must dispatch through AsyncOpenAI.embeddings.create."""
    monkeypatch.setenv("VITA_EMBEDDING_URL", "https://x.example.com/v1")
    monkeypatch.setenv("VITA_EMBEDDING_KEY", "sk-x")

    from vita.memory import rag_memory

    fake_embedding = [0.0, 1.0, 2.0]

    class _FakeResp:
        data = [type("D", (), {"embedding": fake_embedding})()]

    fake_create = AsyncMock(return_value=_FakeResp())

    with patch("openai.resources.embeddings.AsyncEmbeddings.create", fake_create):
        result = asyncio.run(rag_memory._embed_batch_async(["hello"]))

    assert result == [fake_embedding]
    assert fake_create.await_count == 1
