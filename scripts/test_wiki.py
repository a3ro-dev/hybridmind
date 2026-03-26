from datasets import load_dataset
from itertools import islice
from tqdm import tqdm

print("Testing Wikipedia loading...")
try:
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    print("Dataset object created. Iterating...")
    for item in tqdm(islice(ds, 10), total=10):
        print(f"Title: {item['title']}")
except Exception as e:
    print(f"Error: {e}")
