"""Scale RunPod warm workers to zero after the local memory benchmark exits."""

from __future__ import annotations

import datetime
import time

import psutil

try:
    from scripts.runpod_endpoint_admin import set_all_workers_min
except ModuleNotFoundError:  # direct script execution
    from runpod_endpoint_admin import set_all_workers_min


def is_memorybench_running() -> bool:
    for process in psutil.process_iter(("cmdline",)):
        try:
            command = " ".join(process.info.get("cmdline") or ())
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        if "index.ts" in command or "tsx" in command:
            return True
    return False


def main() -> int:
    print("Monitoring memorybench; RunPod workers will scale to zero when it exits.")
    while is_memorybench_running():
        time.sleep(30)

    print(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] scaling down")
    for endpoint_id, name, actual in set_all_workers_min(0):
        print(f"{name} ({endpoint_id}): workersMin={actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
