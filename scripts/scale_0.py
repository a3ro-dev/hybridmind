"""Compatibility command: scale every RunPod endpoint to zero warm workers."""

try:
    from scripts.runpod_endpoint_admin import set_all_workers_min
except ModuleNotFoundError:  # direct script execution
    from runpod_endpoint_admin import set_all_workers_min


def main() -> int:
    for endpoint_id, name, actual in set_all_workers_min(0):
        print(f"{name} ({endpoint_id}): workersMin={actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
