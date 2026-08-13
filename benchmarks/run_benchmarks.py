"""Deprecated unsafe benchmark entry point.

The former script mutated whichever live server/database happened to be at
localhost, used an obsolete 384-dimensional memory estimate, accepted partial
failures, and could therefore emit misleading result files. It is retained as
a fail-closed pointer so old automation cannot silently run that experiment.
"""

from __future__ import annotations

import sys


MESSAGE = """This legacy live benchmark is disabled because it was destructive and
scientifically invalid. Run the bounded zero-provider harness instead:

  .venv\\Scripts\\python.exe scripts\\offline_resource_frontier.py \\
    --output benchmarks\\results\\offline_resource_frontier.json

Then validate a priced, checksum-bound live plan as documented in
docs/RESOURCE_SPEED_TOKENOMICS.md. No result was written.
"""


def main() -> int:
    print(MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
