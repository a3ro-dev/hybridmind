"""List RunPod endpoints without printing credentials or full provider payloads."""

try:
    from scripts.runpod_endpoint_admin import list_endpoints
except ModuleNotFoundError:  # direct script execution
    from runpod_endpoint_admin import list_endpoints


def main() -> int:
    endpoints = list_endpoints()
    print(f"RunPod endpoints: {len(endpoints)}")
    for endpoint in endpoints:
        print(
            f"{endpoint.get('name', '<unnamed>')} ({endpoint.get('id', '<missing>')}): "
            f"workersMin={endpoint.get('workersMin')}, workersMax={endpoint.get('workersMax')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
