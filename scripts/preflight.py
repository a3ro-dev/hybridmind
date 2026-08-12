"""Preflight check for eval runs — verify every remote dependency is live
BEFORE spending LLM tokens on a benchmark. Exit non-zero if anything is down.

Usage:  python scripts/preflight.py
"""
import os
import sys
import time
import httpx

# Load .env (no python-dotenv dependency to avoid frame issues on some setups)
ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(ENV_PATH):
    for line in open(ENV_PATH):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import settings


def check(name, fn):
    try:
        ok, detail = fn()
        mark = "OK " if ok else "DOWN"
        print(f"[{mark}] {name}: {detail}")
        return ok
    except Exception as e:
        print(f"[DOWN] {name}: {type(e).__name__}: {str(e)[:100]}")
        return False


def zai():
    key = settings.zai_api_key
    if not key:
        return False, "ZAI_API_KEY is not set"
    base = settings.zai_base_url.rstrip("/")
    # A tiny deterministic completion verifies authentication and GLM-4.6
    # availability instead of relying on an optional models endpoint.
    r = httpx.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": settings.qa_model, "messages": [{"role": "user", "content": "Reply OK."}], "max_tokens": 4, "temperature": 0},
        timeout=45,
    )
    detail = ""
    if r.status_code != 200:
        detail = f": {r.text[:300]}"
        try:
            models = httpx.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"}, timeout=20)
            if models.status_code == 200:
                ids = [str(item.get("id")) for item in models.json().get("data", [])[:20]]
                detail += f"; available: {', '.join(ids)}"
        except httpx.HTTPError:
            pass
    return r.status_code == 200, f"HTTP {r.status_code}{detail}"


def research_proxy():
    if not settings.allow_research_proxy:
        return False, "disabled (set HYBRIDMIND_ALLOW_RESEARCH_PROXY=true for research only)"
    if not settings.research_proxy_api_key:
        return False, "research proxy enabled but no API key is configured"
    base = settings.research_proxy_base_url.rstrip("/")
    payload = {
        "model": settings.research_proxy_model,
        "messages": [{"role": "user", "content": "Reply OK."}],
        "max_tokens": 4,
        "temperature": 0,
    }
    if "qwen" in settings.research_proxy_model.lower():
        payload["reasoning_effort"] = "none"
    response = httpx.post(
        f"{base}/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.research_proxy_api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=45,
    )
    return response.status_code == 200, f"HTTP {response.status_code} (research-only opt-in)"


def runpod_llm():
    eid = settings.runpod_llm_endpoint_id
    key = settings.runpod_api_key
    if not eid or not key:
        return False, "RunPod LLM credentials/endpoint are not configured"
    r = httpx.get(f"https://api.runpod.ai/v2/{eid}/health",
                  headers={"Authorization": f"Bearer {key}"}, timeout=15)
    j = r.json()
    ready = j.get("workers", {}).get("ready", 0)
    return r.status_code == 200 and ready > 0, f"HTTP {r.status_code}, {ready} workers ready"


def runpod_tei():
    base = os.getenv("RUNPOD_TEI_EMBEDDING_URL", "")
    want = 4096
    if settings.embedding_dimension != want:
        return False, f"configured dimension={settings.embedding_dimension}; HybridMind requires {want}"
    if not base:
        return False, f"RUNPOD_TEI_EMBEDDING_URL not set — no fallback, server will refuse to start with dim={want}"
    key = settings.runpod_api_key
    # Serverless load-balancer endpoint: scales to zero, so the first call
    # after idle pays a cold-start tax (worker waking + 8B model load). That's
    # a state to wait through, not a failure — retry for up to ~3 min. This
    # also WARMS the endpoint so the eval's first real call isn't the cold one.
    h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    # Cold start + GPU scheduling can take minutes; tune via env for scarce-GPU days.
    budget = float(os.getenv("HYBRIDMIND_PREFLIGHT_WARM_SECONDS", "180"))
    deadline = time.monotonic() + budget

    attempt = 0
    while True:
        attempt += 1
        try:
            r = httpx.post(f"{base}/embed", headers=h,
                           json={"inputs": "preflight"}, timeout=60)
            if r.status_code != 200:
                if r.status_code in (408, 429, 500, 502, 503, 504) and time.monotonic() < deadline:
                    time.sleep(5)
                    continue
                return False, f"HTTP {r.status_code}"
            dim = len(r.json()[0]) if isinstance(r.json(), list) else "?"
            note = f"HTTP 200, dim={dim} (expected {want})"
            if attempt > 1:
                note += f" [warm after {attempt} tries]"
            return str(dim) == str(want), note
        except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as e:
            if time.monotonic() >= deadline:
                return False, f"{type(e).__name__} after {attempt} tries (cold start too slow / worker down)"
            time.sleep(3)


if __name__ == "__main__":
    print("=== HybridMind eval preflight ===")
    if settings.allow_research_proxy:
        print("[SKIP] Z.AI GLM-4.6: research proxy explicitly selected; preserving paid budget")
        research_ok = check("Hack Club research proxy (explicit opt-in)", research_proxy)
        # Research mode intentionally selects the proxy and does not fall back
        # to paid Z.AI if the proxy is down.
        hosted_ok = research_ok
    else:
        hosted_ok = check("Z.AI GLM-4.6 (canonical answering/judge)", zai)
    results = [
        hosted_ok,
        check("RunPod LLM (decomposition)", runpod_llm),
        check("RunPod TEI embedding (retrieval, strict 4096-dim)", runpod_tei),
    ]
    if all(results):
        print("\nAll systems go. Safe to run eval.")
        sys.exit(0)
    print("\nFAIL: one or more dependencies are down. Do NOT run eval — "
          "fix the DOWN item(s) first (e.g. start the RunPod pod).")
    sys.exit(1)
