"""Set warm workers for explicitly configured HybridMind RunPod endpoints."""

from __future__ import annotations

import argparse

try:
    from scripts.runpod_endpoint_admin import configured_endpoint_ids, set_workers_min
except ModuleNotFoundError:  # direct script execution
    from runpod_endpoint_admin import configured_endpoint_ids, set_workers_min


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workers_min", type=int)
    parser.add_argument("endpoint_ids", nargs="*")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    endpoint_ids = args.endpoint_ids or configured_endpoint_ids()
    if not endpoint_ids:
        raise SystemExit(
            "No endpoint IDs supplied; configure RUNPOD_TEI_ENDPOINT_ID or "
            "RUNPOD_LLM_ENDPOINT_ID"
        )
    for endpoint_id in endpoint_ids:
        result = set_workers_min(endpoint_id, args.workers_min)
        print(
            f"{result.get('name', endpoint_id)} ({endpoint_id}): "
            f"workersMin={result.get('workersMin', args.workers_min)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
