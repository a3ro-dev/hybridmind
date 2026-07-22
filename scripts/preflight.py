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


def check(name, fn):
    try:
        ok, detail = fn()
        mark = "OK " if ok else "DOWN"
        print(f"[{mark}] {name}: {detail}")
        return ok
    except Exception as e:
        print(f"[DOWN] {name}: {type(e).__name__}: {str(e)[:100]}")
        return False


def hc():
    key = os.getenv("HC_API_KEY")
    r = httpx.get("https://ai.hackclub.com/proxy/v1/models",
                  headers={"Authorization": f"Bearer {key}"}, timeout=15)
    return r.status_code == 200, f"HTTP {r.status_code}"


def runpod_llm():
    eid = os.getenv("RUNPOD_LLM_ENDPOINT_ID", "")
    key = os.getenv("RUNPOD_API_KEY")
    r = httpx.get(f"https://api.runpod.ai/v2/{eid}/health",
                  headers={"Authorization": f"Bearer {key}"}, timeout=15)
    j = r.json()
    ready = j.get("workers", {}).get("ready", 0)
    return r.status_code == 200 and ready > 0, f"HTTP {r.status_code}, {ready} workers ready"


def runpod_tei():
    base = os.getenv("RUNPOD_TEI_EMBEDDING_URL", "")
    want = os.getenv("HYBRIDMIND_EMBEDDING_DIMENSION", "1024")
    if not base:
        return True, f"Local/remote fallback active, dim={want}"
    key = os.getenv("RUNPOD_API_KEY")
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
    results = [
        check("HackClub proxy (answering/judge)", hc),
        check("RunPod LLM (decomposition)", runpod_llm),
        check("RunPod TEI embedding (retrieval)", runpod_tei),
    ]
    if all(results):
        print("\nAll systems go. Safe to run eval.")
        sys.exit(0)
    print("\nFAIL: one or more dependencies are down. Do NOT run eval — "
          "fix the DOWN item(s) first (e.g. start the RunPod pod).")
    sys.exit(1)
