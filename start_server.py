"""Start HybridMind API server as a background process."""
import subprocess
import sys
import os

os.chdir(r"D:\hybridmind")

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

log = open(r"D:\hybridmind\server.log", "w")
proc = subprocess.Popen(
    [sys.executable, "-u", "main.py"],
    stdout=log,
    stderr=log,
    env=env,
    cwd=r"D:\hybridmind",
    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
)
print(f"Server PID: {proc.pid}")
with open(r"D:\hybridmind\server.pid", "w") as f:
    f.write(str(proc.pid))

