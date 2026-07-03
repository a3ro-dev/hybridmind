"""
Download the MuSiQue dev set from HuggingFace.

Usage:
    python benchmarks/data/musique/download_musique.py
"""
import gzip
import shutil
import sys
import urllib.request
from pathlib import Path

DEST_DIR = Path(__file__).parent
JSONL_FILE = DEST_DIR / "musique_ans_v1.0_dev.jsonl"
GZ_URL = (
    "https://huggingface.co/datasets/allenai/musique/resolve/main/data/"
    "musique_ans_v1.0_dev.jsonl.gz"
)


def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    if JSONL_FILE.exists():
        print(f"Already exists: {JSONL_FILE}  ({JSONL_FILE.stat().st_size / 1e6:.1f} MB)")
        return

    gz_path = DEST_DIR / "musique_ans_v1.0_dev.jsonl.gz"
    print(f"Downloading {GZ_URL} ...")
    try:
        urllib.request.urlretrieve(GZ_URL, gz_path)
    except Exception as e:
        print(f"ERROR downloading: {e}")
        print("Manual download:")
        print(f"  1. Go to https://huggingface.co/datasets/allenai/musique/tree/main/data")
        print(f"  2. Download musique_ans_v1.0_dev.jsonl.gz")
        print(f"  3. Extract it to {DEST_DIR}/")
        sys.exit(1)

    print("Extracting...")
    with gzip.open(gz_path, "rb") as f_in, open(JSONL_FILE, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()

    lines = sum(1 for _ in open(JSONL_FILE))
    print(f"Done: {JSONL_FILE}  ({lines} questions, {JSONL_FILE.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
