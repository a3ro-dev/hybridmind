"""Compatibility command: keep one warm worker on every RunPod endpoint."""

try:
    from scripts.runpod_endpoint_admin import set_all_workers_min
except ModuleNotFoundError:  # direct ``python scripts/set_min_workers.py`` execution
    from runpod_endpoint_admin import set_all_workers_min


def main() -> int:
    for endpoint_id, name, actual in set_all_workers_min(1):
        print(f"{name} ({endpoint_id}): workersMin={actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
