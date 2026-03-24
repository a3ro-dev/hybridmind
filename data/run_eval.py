import json
import httpx

def main():
    print("Loading eval queries...")
    with open("data/eval_queries.json") as f:
        eval_set = json.load(f)
    print(f"Loaded {len(eval_set['queries'])} queries")

    print("Requesting comparison/effectiveness...")
    r = httpx.post(
        "http://localhost:8000/comparison/effectiveness", 
        json=eval_set, 
        timeout=120.0
    )
    
    if r.status_code == 200:
        print("Success! Metrics:")
        print(json.dumps(r.json(), indent=2))
    else:
        print(f"Failed ({r.status_code}): {r.text}")

if __name__ == "__main__":
    main()
