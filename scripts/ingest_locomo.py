"""Ingest LoCoMo benchmark data into HybridMind for retrieval eval.

Conversation turns are written in dataset order so ``next_turn`` edges are
deterministic.  Every turn carries a stable, conversation-qualified LoCoMo
evidence ID used by exact-evidence retrieval evaluation.
"""
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval_common import api_headers, sanitized_error

LOCOMO_PATH = Path("memorybench/data/benchmarks/locomo/locomo10.json")
BASE_URL = os.getenv("HYBRIDMIND_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
# Turn creation is deliberately serial: the API derives ``next_turn`` from
# insertion order, so parallel writes would make graph edges nondeterministic.
CONCURRENCY = 1
TIMEOUT = 300.0
EXTRACT_FACTS = os.getenv("HYBRIDMIND_LOCOMO_EXTRACT_FACTS", "false").lower() == "true"


def evidence_id(sample_id: str, dia_id: str) -> str:
    return f"locomo:{sample_id}:{dia_id}"


def normalize_locomo_event_time(value: object) -> str | None:
    """Convert the dataset's session timestamp to explicit UTC ISO-8601.

    LoCoMo records values such as ``1:56 pm on 8 May, 2023`` without a
    timezone.  We preserve that raw value in metadata and make the UTC
    assumption explicit for ordering; malformed non-empty values fail closed.
    """
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(text, "%I:%M %p on %d %B, %Y")
        except ValueError as exc:
            raise ValueError(
                "LoCoMo session timestamp must be ISO-8601 or match the "
                "canonical dataset format"
            ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.isoformat()


def _session_sort_key(name: str) -> tuple[int, str]:
    try:
        return int(name.removeprefix("session_")), name
    except ValueError:
        return sys.maxsize, name


def _matches_evidence(result: dict, *, expected_id: str, expected_text: str) -> bool:
    metadata = result.get("metadata") or {}
    return metadata.get("evidence_id") == expected_id and result.get("text") == expected_text


async def evidence_exists(
    client: httpx.AsyncClient, *, expected_id: str, expected_text: str
) -> bool:
    """Verify an evidence item through the configured API, local or remote.

    The public node-create API does not accept caller-selected node IDs.  Resume
    therefore uses the stable ``evidence_id`` metadata key and additionally
    verifies the full stored text, so an ID collision cannot silently skip a
    different item.
    """
    response = await client.post(
        "/search/vector",
        json={
            "query_text": expected_text,
            "top_k": 10,
            "min_score": 0.0,
            "filter_metadata": {"evidence_id": expected_id},
        },
    )
    response.raise_for_status()
    body = response.json()
    results = body.get("results") if isinstance(body, dict) else None
    if not isinstance(results, list):
        raise RuntimeError("evidence verification returned an invalid response schema")
    return any(
        isinstance(result, dict)
        and _matches_evidence(
            result, expected_id=expected_id, expected_text=expected_text
        )
        for result in results
    )


async def post_node(client: httpx.AsyncClient, sem: asyncio.Semaphore, payload: dict) -> bool:
    """POST one node with API-scoped idempotency checks and bounded retries."""
    expected_id = str((payload.get("metadata") or {}).get("evidence_id") or "")
    expected_text = str(payload.get("text") or "")
    if not expected_id or not expected_text:
        raise ValueError("LoCoMo nodes require non-empty evidence_id and text")

    async with sem:
        if await evidence_exists(
            client, expected_id=expected_id, expected_text=expected_text
        ):
            return True

        for attempt in range(6):
            try:
                resp = await client.post("/nodes", json=payload)
                if resp.status_code in (200, 201):
                    body = resp.json()
                    if not isinstance(body, dict) or not _matches_evidence(
                        body, expected_id=expected_id, expected_text=expected_text
                    ):
                        raise RuntimeError("node-create response failed evidence verification")
                    return True

                # A future API may enforce uniqueness and return 409.  Accept
                # it only after verifying the remotely stored metadata/text.
                if resp.status_code == 409 and await evidence_exists(
                    client, expected_id=expected_id, expected_text=expected_text
                ):
                    return True
                if resp.status_code != 409:
                    try:
                        if await evidence_exists(
                            client, expected_id=expected_id, expected_text=expected_text
                        ):
                            return True
                    except Exception as verification_error:
                        print(
                            f"  [attempt {attempt + 1}] create outcome could not be "
                            f"verified: {sanitized_error(verification_error)}",
                            flush=True,
                        )
                        return False
                print(
                    f"  [attempt {attempt + 1}] node create HTTP {resp.status_code}",
                    flush=True,
                )
            except Exception as e:
                # If the response was lost after a successful commit, verify
                # through the API before considering another POST.
                try:
                    if await evidence_exists(
                        client, expected_id=expected_id, expected_text=expected_text
                    ):
                        return True
                except Exception as verification_error:
                    print(
                        f"  [attempt {attempt + 1}] create outcome could not be "
                        f"verified: {sanitized_error(verification_error)}",
                        flush=True,
                    )
                    return False
                print(
                    f"  [attempt {attempt + 1}] {sanitized_error(e)}",
                    flush=True,
                )
            await asyncio.sleep(1.5 * (attempt + 1))
        return False


async def ingest_session(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    sample_id: str,
    all_turns: list,
    speakers: dict,
    idx: int,
    total: int,
) -> tuple[int, int]:
    """Ingest one conversation in order and optionally extract facts."""
    ok = 0
    failed = 0
    for turn in all_turns:
        role = "ai" if turn["speaker"].startswith(speakers.get("speaker_b", "ai")[:3]) else "human"
        date_str = turn["date"]
        event_time = normalize_locomo_event_time(date_str)
        txt = turn["text"]
        text_with_meta = (
            f"[DATE: {date_str}] [SPEAKER: {turn['speaker']}] {txt}"
            if date_str else
            f"[SPEAKER: {turn['speaker']}] {txt}"
        )
        payload = {
            "text": text_with_meta,
            "metadata": {
                "session_id": sample_id,
                "sessionId": sample_id,
                "containerTag": "locomo",
                "benchmark": "locomo",
                "benchmark_sample_id": sample_id,
                "role": role,
                "timestamp": date_str,
                "event_time": event_time,
                "event_time_source": "locomo_session_date_time",
                "event_time_timezone_assumption": "dataset_unspecified_assumed_utc",
                "dia_id": turn["dia_id"],
                "evidence_id": turn["evidence_id"],
                # Held-out conversation-split evidence (2026-08-14) showed
                # speaker-prefixed sparse keys improve exact source recall at
                # a bounded footprint. Keep the authoritative/dense text and
                # temporal metadata unchanged; only the sparse representation
                # uses this source-grounded alternate key.
                "sparse_text": f"{turn['speaker']}: {txt}",
                "locomo_session": turn["session_name"],
                "turn_index": turn["turn_index"],
            },
            "event_time": event_time,
        }
        if await post_node(client, sem, payload):
            ok += 1
        else:
            failed += 1
    print(f"  [{idx}/{total}] {sample_id}: {ok}/{len(all_turns)} turns ingested"
          + (f" ({failed} failed)" if failed else ""), flush=True)

    if failed:
        return ok, failed

    if EXTRACT_FACTS:
        resp = await client.post("/ingest/session-facts", json={
            "session_id": sample_id,
            "turns": all_turns,
            "container_tag": "locomo",
        })
        resp.raise_for_status()
        fj = resp.json()
        if "facts_extracted" not in fj:
            raise RuntimeError(f"fact extraction returned no status for {sample_id}")
        nf = fj.get("facts_extracted", 0)
        print(f"  [{idx}/{total}] {sample_id}: {nf} facts extracted", flush=True)
        ok += nf

    return ok, 0


async def main():
    # Health check
    async with httpx.AsyncClient(
        base_url=BASE_URL, timeout=30.0, headers=api_headers()
    ) as c:
        try:
            # /health performs a real remote embedding and can block the single
            # API worker during a serverless cold start. Readiness only verifies
            # that the initialized engine and indexes are available.
            r = await c.get("/ready")
            r.raise_for_status()
            print("HybridMind is healthy", flush=True)
        except Exception as e:
            print(f"HybridMind not reachable: {sanitized_error(e)}")
            sys.exit(1)

    data = json.loads(LOCOMO_PATH.read_text())
    print(f"Loaded {len(data)} conversations", flush=True)

    sem = asyncio.Semaphore(CONCURRENCY)
    total_nodes = 0

    async with httpx.AsyncClient(
        base_url=BASE_URL,
        timeout=TIMEOUT,
        headers=api_headers(),
        limits=httpx.Limits(max_connections=CONCURRENCY + 4, max_keepalive_connections=CONCURRENCY),
    ) as client:
        for i, conv in enumerate(data, 1):
            sample_id = conv.get("sample_id", f"locomo_{i}")

            convo = conv.get("conversation", {})
            speakers = {
                "speaker_a": convo.get("speaker_a", "human"),
                "speaker_b": convo.get("speaker_b", "ai"),
            }

            all_turns = []
            for key in sorted(convo.keys(), key=_session_sort_key):
                if key.startswith("session_") and not key.endswith("_date_time"):
                    date = convo.get(key + "_date_time", "")
                    messages = convo[key]
                    if not isinstance(messages, list):
                        continue
                    for turn_index, msg in enumerate(messages):
                        if isinstance(msg, dict) and "text" in msg:
                            txt = msg["text"].strip()
                            if txt:
                                dia_id = str(msg.get("dia_id") or f"{key}:{turn_index}")
                                all_turns.append({
                                    "speaker": msg.get("speaker", "unknown"),
                                    "text": txt,
                                    "date": date,
                                    "session_name": key,
                                    "turn_index": turn_index,
                                    "dia_id": dia_id,
                                    "evidence_id": evidence_id(sample_id, dia_id),
                                })

            if not all_turns:
                print(f"  [{i}/{len(data)}] {sample_id}: no turns, skipping", flush=True)
                continue

            # ``post_node`` checks every stable evidence ID through BASE_URL,
            # so resume works identically for local and remote deployments.
            n, failed = await ingest_session(
                client, sem, sample_id, all_turns, speakers, i, len(data)
            )
            total_nodes += n
            if failed:
                raise RuntimeError(
                    f"LoCoMo ingestion failed closed: {failed} turns failed for {sample_id}"
                )

    print(f"\nDone: {total_nodes} nodes across {len(data)} sessions", flush=True)
    print("Ready: python eval_locomo_retrieval.py")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"LoCoMo ingestion failed: {sanitized_error(exc)}", flush=True)
        raise SystemExit(1)
