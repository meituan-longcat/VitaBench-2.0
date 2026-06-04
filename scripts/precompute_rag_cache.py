"""
Offline precompute of RAG embeddings for the personalization benchmark.

Produces one .npz per (user_id, subtask_id) under --out-dir; the
RAGCacheMemory backend loads these at benchmark time instead of calling
the embedding API.

Usage:

    python3 scripts/precompute_rag_cache.py \\
        --tasks-file data/vita/domains/personalization/tasks.json \\
        --out-dir    data/vita/domains/personalization/rag_cache \\
        --concurrency 16 --rpm 800

The embedding endpoint is the same one RAGMemory uses at runtime — pulled
from vita.models.yaml `default.base_url` or overridden via VITA_EMBEDDING_URL
/ VITA_EMBEDDING_KEY / VITA_EMBEDDING_MODEL. Concurrency and an RPM
token-bucket cap keep the endpoint stable; exponential back-off handles
transient 429/5xx.

Restart / resume semantics:
  - Already-complete .npz files whose `model`, `chunk_size`, `chunk_overlap`
    match the requested settings are skipped. Pass --force to overwrite.
  - Subtasks that fail after all retries are listed in {out-dir}/retry.txt;
    pass --resume-from retry.txt to re-run only those.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger
from openai import AsyncOpenAI
from tqdm import tqdm

# Make `vita` importable when invoked as `python3 scripts/...` from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from vita.memory.rag_memory import (  # noqa: E402
    RAGMemory,
    _embed_client,
    _split_text_by_tokens,
)


# ─── Rate-limited embedding pool ─────────────────────────────────────────────


class TokenBucket:
    """Minimal async token bucket for RPM control.

    `capacity` is `rpm`, refilled uniformly at `rpm / 60` tokens per second.
    Each embedding request consumes one token.
    """

    def __init__(self, rpm: int):
        self.rpm = rpm
        self.capacity = float(rpm)
        self._tokens = float(rpm)
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._last
                self._tokens = min(
                    self.capacity, self._tokens + elapsed * (self.rpm / 60.0)
                )
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # Wait for the next refill that produces at least 1 token.
                wait = (1.0 - self._tokens) * (60.0 / self.rpm)
                await asyncio.sleep(wait)


async def _embed_one(
    client: AsyncOpenAI,
    model: str,
    text: str,
    *,
    sem: asyncio.Semaphore,
    bucket: TokenBucket,
    max_retries: int = 6,
) -> Optional[List[float]]:
    """Embed one text with exponential back-off and RPM control.

    Returns None when all retries are exhausted. Caller substitutes a
    zero-vector placeholder so the whole subtask cache still gets
    persisted — a later `--retry-zeros` run can fill in the gaps.
    """
    delay = 1.0
    last_err: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        await bucket.acquire()
        async with sem:
            try:
                resp = await client.embeddings.create(
                    model=model, input=text.replace("\n", " ")
                )
                return resp.data[0].embedding
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt == max_retries:
                    logger.warning(
                        f"Embed exhausted {max_retries + 1} attempts ({e!r}); "
                        f"returning zero-vector placeholder"
                    )
                    return None
                logger.warning(
                    f"Embed attempt {attempt + 1} failed ({e!r}); retrying in {delay:.1f}s"
                )
        await asyncio.sleep(delay)
        delay = min(delay * 2, 60.0)
    # Defensive — loop always returns inside the for body.
    logger.warning(f"Embed fell through retry loop (last_err={last_err!r})")
    return None


# ─── Subtask -> chunks rendering (reuses RAGMemory helpers) ──────────────────


def _render_subtask_chunks(
    interactions: list, chunk_size: int, chunk_overlap: int
) -> list[dict]:
    """Run the same chunking RAGMemory.update() does, without embedding.

    Output is a list of dicts with keys: text, keywords, timestamp, type.
    """
    base_chunks = [RAGMemory._interaction_to_chunk(i) for i in interactions]
    sub_chunks: list[dict] = []
    for base in base_chunks:
        pieces = _split_text_by_tokens(base["text"], chunk_size, chunk_overlap)
        if not pieces:
            continue
        for piece in pieces:
            sub_chunks.append(
                {
                    "text": piece,
                    "keywords": list(base.get("keywords") or []),
                    "timestamp": base.get("timestamp", ""),
                    "type": base.get("type", "unknown"),
                }
            )
    return sub_chunks


# ─── Cache file read / write ─────────────────────────────────────────────────


def _cache_path(out_dir: Path, user_id: str, subtask_id: str) -> Path:
    return out_dir / f"{user_id}__{subtask_id}.npz"


def _cache_is_valid(
    path: Path, model: str, chunk_size: int, chunk_overlap: int
) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=True) as data:
            if str(data.get("model", "")) != model:
                return False
            if int(data.get("chunk_size", -1)) != chunk_size:
                return False
            if int(data.get("chunk_overlap", -1)) != chunk_overlap:
                return False
            # Presence of core arrays
            _ = data["chunk_embeddings"]
            _ = data["chunk_texts"]
            _ = data["query_embedding"]
            _ = data["instruction"]
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Rejecting cache {path}: {e!r}")
        return False
    return True


def _write_cache(
    path: Path,
    sub_chunks: list[dict],
    chunk_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    instruction: str,
    model: str,
    chunk_size: int,
    chunk_overlap: int,
    chunk_zero_mask: np.ndarray,
    query_is_zero: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # np.savez_compressed auto-appends ".npz" if the path doesn't already end
    # in ".npz", so we give it a full ".tmp.npz" path and rename that onto the
    # final path atomically.
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        str(tmp),
        chunk_embeddings=chunk_embeddings.astype(np.float32),
        chunk_texts=np.asarray([c["text"] for c in sub_chunks], dtype=object),
        chunk_keywords=np.asarray([c["keywords"] for c in sub_chunks], dtype=object),
        chunk_timestamps=np.asarray([c["timestamp"] for c in sub_chunks], dtype=object),
        chunk_types=np.asarray([c["type"] for c in sub_chunks], dtype=object),
        query_embedding=query_embedding.astype(np.float32),
        instruction=instruction,
        model=model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_zero_mask=chunk_zero_mask.astype(bool),
        query_is_zero=bool(query_is_zero),
    )
    tmp.replace(path)


# ─── Per-subtask worker ──────────────────────────────────────────────────────


async def _process_subtask(
    client: AsyncOpenAI,
    model: str,
    user_id: str,
    subtask: dict,
    out_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    sem: asyncio.Semaphore,
    bucket: TokenBucket,
    force: bool,
) -> Tuple[str, str, str]:
    """Process one subtask. Returns (status, user_id, subtask_id).

    status ∈ {"ok", "skipped", "empty", "failed:<reason>"}
    """
    subtask_id = str(subtask.get("subtask_id", ""))
    # Outer catch — any exception here (I/O, malformed subtask, embed
    # failure, etc.) is converted to a "failed:..." return so one bad
    # subtask can't tear down the whole run via the as_completed loop.
    try:
        instruction = subtask.get("instruction", "") or ""
        interactions = subtask.get("interactions") or []

        path = _cache_path(out_dir, user_id, subtask_id)
        if not force and _cache_is_valid(path, model, chunk_size, chunk_overlap):
            return ("skipped", user_id, subtask_id)

        if not interactions and not instruction:
            return ("empty", user_id, subtask_id)

        sub_chunks = _render_subtask_chunks(interactions, chunk_size, chunk_overlap)

        # Embed: chunk texts first (can be 0), then query text last.
        texts = [c["text"] for c in sub_chunks]
        texts.append(instruction)

        # return_exceptions=True so one bad request can't poison the gather.
        # _embed_one itself returns None on exhaustion, but we also catch any
        # escaped exception and treat it the same (zero-vector placeholder).
        raw = await asyncio.gather(
            *[
                _embed_one(client, model, t, sem=sem, bucket=bucket)
                for t in texts
            ],
            return_exceptions=True,
        )

        # Determine embedding dim: prefer any successful response; fall back
        # to 3072 (text-embedding-3-large) if every single request failed.
        dim: Optional[int] = None
        for r in raw:
            if isinstance(r, list):
                dim = len(r)
                break
        if dim is None:
            dim = 3072

        embs: list[list[float]] = []
        is_zero: list[bool] = []
        for r in raw:
            if isinstance(r, list):
                embs.append(r)
                is_zero.append(False)
            else:
                if isinstance(r, BaseException):
                    logger.warning(
                        f"{user_id}__{subtask_id}: embed raised ({r!r}); using zero-vector"
                    )
                embs.append([0.0] * dim)
                is_zero.append(True)

        query_emb = np.asarray(embs[-1], dtype=np.float32)
        query_is_zero = is_zero[-1]

        if sub_chunks:
            chunk_embs = np.asarray(embs[:-1], dtype=np.float32)
            chunk_zero_mask = np.asarray(is_zero[:-1], dtype=bool)
        else:
            chunk_embs = np.zeros((0, dim), dtype=np.float32)
            chunk_zero_mask = np.zeros((0,), dtype=bool)

        _write_cache(
            path,
            sub_chunks=sub_chunks,
            chunk_embeddings=chunk_embs,
            query_embedding=query_emb,
            instruction=instruction,
            model=model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_zero_mask=chunk_zero_mask,
            query_is_zero=query_is_zero,
        )

        n_zero = int(chunk_zero_mask.sum()) + int(query_is_zero)
        if n_zero:
            return (f"ok_with_zeros:{n_zero}", user_id, subtask_id)
        return ("ok", user_id, subtask_id)
    except Exception as e:  # noqa: BLE001
        return (f"failed:{e!r}", user_id, subtask_id)


# ─── Retry-zeros worker ──────────────────────────────────────────────────────


async def _retry_zeros_one(
    client: AsyncOpenAI,
    model: str,
    path: Path,
    sem: asyncio.Semaphore,
    bucket: TokenBucket,
) -> Tuple[str, str]:
    """Re-embed only the zero-placeholder rows in one .npz, rewrite it.

    Returns (status, path_name).  status ∈ {"unchanged", "filled:N", "still:N",
    "failed:<reason>"}
    """
    try:
        with np.load(path, allow_pickle=True) as data:
            chunk_embeddings = data["chunk_embeddings"].astype(np.float32).copy()
            chunk_texts = list(data["chunk_texts"])
            chunk_keywords = data["chunk_keywords"]
            chunk_timestamps = data["chunk_timestamps"]
            chunk_types = data["chunk_types"]
            query_embedding = data["query_embedding"].astype(np.float32).copy()
            instruction = str(data["instruction"])
            chunk_size = int(data["chunk_size"])
            chunk_overlap = int(data["chunk_overlap"])
            cached_model = str(data["model"])
            if "chunk_zero_mask" in data.files:
                chunk_zero_mask = data["chunk_zero_mask"].astype(bool).copy()
            else:
                chunk_zero_mask = np.zeros(len(chunk_texts), dtype=bool)
            query_is_zero = bool(data["query_is_zero"]) if "query_is_zero" in data.files else False

        if cached_model != model:
            return (f"failed:model-mismatch({cached_model}!={model})", path.name)

        zero_chunk_idx = [int(i) for i, z in enumerate(chunk_zero_mask) if z]
        if not zero_chunk_idx and not query_is_zero:
            return ("unchanged", path.name)

        # Build the tasks: each zero chunk + optionally the query.
        targets = [(i, str(chunk_texts[i])) for i in zero_chunk_idx]
        if query_is_zero:
            targets.append((-1, instruction))  # -1 marks the query slot

        raw = await asyncio.gather(
            *[
                _embed_one(client, model, t, sem=sem, bucket=bucket)
                for _, t in targets
            ],
            return_exceptions=True,
        )

        filled = 0
        still = 0
        for (slot, _), r in zip(targets, raw):
            if isinstance(r, list):
                vec = np.asarray(r, dtype=np.float32)
                if slot == -1:
                    query_embedding = vec
                    query_is_zero = False
                else:
                    chunk_embeddings[slot] = vec
                    chunk_zero_mask[slot] = False
                filled += 1
            else:
                still += 1

        # Rebuild sub_chunks list in the _write_cache expected shape.
        sub_chunks = [
            {
                "text": str(chunk_texts[i]),
                "keywords": list(chunk_keywords[i]),
                "timestamp": str(chunk_timestamps[i]),
                "type": str(chunk_types[i]),
            }
            for i in range(len(chunk_texts))
        ]
        _write_cache(
            path,
            sub_chunks=sub_chunks,
            chunk_embeddings=chunk_embeddings,
            query_embedding=query_embedding,
            instruction=instruction,
            model=model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_zero_mask=chunk_zero_mask,
            query_is_zero=query_is_zero,
        )

        if still:
            return (f"still:{still}", path.name)
        return (f"filled:{filled}", path.name)
    except Exception as e:  # noqa: BLE001
        return (f"failed:{e!r}", path.name)


async def _retry_zeros_main(
    client: AsyncOpenAI,
    model: str,
    out_dir: Path,
    concurrency: int,
    rpm: int,
) -> int:
    """Scan out_dir for .npz files with zero-placeholders and re-embed those."""
    paths = sorted(out_dir.glob("*.npz"))
    if not paths:
        print(f"No .npz files under {out_dir}")
        return 0

    # Pre-filter to files that actually have zeros; avoids opening the
    # network session when there's nothing to do.
    todo: list[Path] = []
    for p in paths:
        try:
            with np.load(p, allow_pickle=True) as data:
                has_chunk_zeros = (
                    "chunk_zero_mask" in data.files
                    and bool(data["chunk_zero_mask"].any())
                )
                has_query_zero = (
                    "query_is_zero" in data.files and bool(data["query_is_zero"])
                )
            if has_chunk_zeros or has_query_zero:
                todo.append(p)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Skipping unreadable cache {p}: {e!r}")
    logger.info(f"retry-zeros: {len(todo)} / {len(paths)} cache files have placeholders")
    if not todo:
        return 0

    sem = asyncio.Semaphore(concurrency)
    bucket = TokenBucket(rpm=rpm)

    filled_files = still_files = failed_files = 0
    coros = [
        _retry_zeros_one(client, model, p, sem, bucket)
        for p in todo
    ]
    pbar = tqdm(total=len(coros), desc="retry-zeros")
    for fut in asyncio.as_completed(coros):
        try:
            status, name = await fut
        except Exception as e:  # noqa: BLE001
            failed_files += 1
            pbar.write(f"[RETRY unhandled] {e!r}")
            pbar.update(1)
            continue
        pbar.update(1)
        if status.startswith("filled:"):
            filled_files += 1
        elif status.startswith("still:"):
            still_files += 1
            pbar.write(f"[STILL ZEROS] {name}: {status}")
        elif status == "unchanged":
            pass
        else:
            failed_files += 1
            pbar.write(f"[RETRY FAILED] {name}: {status}")
    pbar.close()

    print(
        f"\nretry-zeros done. fully_filled={filled_files} still_has_zeros={still_files} "
        f"failed={failed_files}  out_dir={out_dir}"
    )
    return 0 if failed_files == 0 and still_files == 0 else 1


# ─── Task iteration ──────────────────────────────────────────────────────────


def _iter_subtasks(
    tasks_data: list,
    *,
    limit_users: Optional[int] = None,
    user_ids: Optional[set] = None,
    resume_keys: Optional[set] = None,
):
    """Yield (user_id, subtask_dict) from tasks.json."""
    count_users = 0
    for user in tasks_data:
        uid = str(user.get("user_id") or user.get("id") or "")
        if user_ids is not None and uid not in user_ids:
            continue
        if limit_users is not None and count_users >= limit_users:
            break
        count_users += 1
        for st in user.get("subtasks", []):
            if resume_keys is not None:
                key = f"{uid}__{st.get('subtask_id', '')}"
                if key not in resume_keys:
                    continue
            yield uid, st


# ─── Main ────────────────────────────────────────────────────────────────────


async def _amain(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client, model_default = _embed_client()
    # Precompute always uses text-embedding-3-large unless overridden.
    model = args.model or model_default
    logger.info(f"Embedding endpoint: {client.base_url}  model={model}")

    if args.retry_zeros:
        return await _retry_zeros_main(
            client, model, out_dir, args.concurrency, args.rpm
        )

    tasks_path = Path(args.tasks_file)
    with open(tasks_path) as f:
        tasks_data = json.load(f)

    resume_keys: Optional[set] = None
    if args.resume_from:
        with open(args.resume_from) as f:
            resume_keys = {line.strip() for line in f if line.strip()}
        logger.info(f"Resume mode: {len(resume_keys)} (user, subtask) keys to retry")

    user_ids = set(args.user_ids) if args.user_ids else None

    subtasks = list(
        _iter_subtasks(
            tasks_data,
            limit_users=args.limit_users,
            user_ids=user_ids,
            resume_keys=resume_keys,
        )
    )
    logger.info(
        f"Precomputing embeddings for {len(subtasks)} subtasks "
        f"(concurrency={args.concurrency}, rpm={args.rpm})"
    )

    sem = asyncio.Semaphore(args.concurrency)
    bucket = TokenBucket(rpm=args.rpm)

    ok = skipped = empty = failed = 0
    ok_with_zeros = 0
    total_zero_embeds = 0
    fail_keys: list[str] = []
    partial_keys: list[str] = []

    coros = [
        _process_subtask(
            client,
            model,
            user_id,
            st,
            out_dir,
            args.chunk_size,
            args.chunk_overlap,
            sem,
            bucket,
            args.force,
        )
        for user_id, st in subtasks
    ]
    pbar = tqdm(total=len(coros), desc="subtasks")
    for fut in asyncio.as_completed(coros):
        try:
            status, uid, sid = await fut
        except Exception as e:  # noqa: BLE001
            failed += 1
            fail_keys.append(f"unknown__{e!r}")
            pbar.update(1)
            pbar.write(f"[FAILED unhandled] {e!r}")
            continue
        pbar.update(1)
        if status == "ok":
            ok += 1
        elif status.startswith("ok_with_zeros:"):
            ok_with_zeros += 1
            ok += 1  # cache is still written; count as ok
            try:
                n_zero = int(status.split(":", 1)[1])
            except ValueError:
                n_zero = 0
            total_zero_embeds += n_zero
            partial_keys.append(f"{uid}__{sid}")
            pbar.write(f"[PARTIAL] {uid}__{sid}: {n_zero} zero-vector placeholders")
        elif status == "skipped":
            skipped += 1
        elif status == "empty":
            empty += 1
        else:
            failed += 1
            fail_keys.append(f"{uid}__{sid}")
            pbar.write(f"[FAILED] {uid}__{sid}: {status}")
    pbar.close()

    print(
        f"\nDone. ok={ok} (of which partial={ok_with_zeros}, "
        f"zero_embeds={total_zero_embeds}) skipped={skipped} empty={empty} "
        f"failed={failed}  out_dir={out_dir}"
    )
    if fail_keys:
        retry_path = out_dir / "retry.txt"
        retry_path.write_text("\n".join(fail_keys) + "\n")
        print(f"Failed keys written to {retry_path}")
    if partial_keys:
        partial_path = out_dir / "partial.txt"
        partial_path.write_text("\n".join(partial_keys) + "\n")
        print(
            f"Partial (zero-vector) keys written to {partial_path}; "
            f"re-fill with: --retry-zeros"
        )
    if fail_keys:
        return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tasks-file",
        default="data/vita/domains/personalization/tasks.json",
        help="Path to tasks.json",
    )
    p.add_argument(
        "--out-dir",
        default="data/vita/domains/personalization/rag_cache",
        help="Where to write <user_id>__<subtask_id>.npz files",
    )
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument(
        "--rpm",
        type=int,
        default=800,
        help="Total embedding requests per minute. User-confirmed online ceiling "
        "is ~1200; 800 leaves headroom for bursts.",
    )
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--chunk-overlap", type=int, default=0)
    p.add_argument("--model", default=None, help="Override embedding model name")
    p.add_argument("--force", action="store_true", help="Overwrite existing cache files")
    p.add_argument(
        "--limit-users",
        type=int,
        default=None,
        help="Only process the first N users (pilot mode)",
    )
    p.add_argument(
        "--user-ids",
        nargs="+",
        default=None,
        help="Only process these user_ids (space-separated)",
    )
    p.add_argument(
        "--resume-from",
        default=None,
        help="Path to a retry.txt listing `<user>__<subtask>` keys to reprocess",
    )
    p.add_argument(
        "--retry-zeros",
        action="store_true",
        help="Scan --out-dir for .npz files containing zero-vector placeholders "
        "(from a prior run that exhausted retries) and re-embed only those "
        "rows. Rewrites each file in place; --tasks-file is not needed.",
    )
    args = p.parse_args()

    if os.environ.get("PYTHONUNBUFFERED") != "1":
        os.environ["PYTHONUNBUFFERED"] = "1"

    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
