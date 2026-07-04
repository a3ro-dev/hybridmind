"""Run all tests and save results."""
import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent

result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
    capture_output=True,
    text=True,
    cwd=str(repo_root),
)

with open(repo_root / "test_results.txt", "w", encoding="utf-8") as f:
    f.write("=== STDOUT ===\n")
    f.write(result.stdout)
    f.write("\n=== STDERR ===\n")
    f.write(result.stderr)
    f.write(f"\n=== EXIT CODE: {result.returncode} ===\n")

print(f"Results saved. Exit code: {result.returncode}")
