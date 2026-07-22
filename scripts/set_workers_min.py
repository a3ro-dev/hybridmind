import os
import sys
import dotenv
import httpx

dotenv.load_dotenv()
key = os.getenv("RUNPOD_API_KEY")
target = int(sys.argv[1]) if len(sys.argv) > 1 else 1

query = """
mutation SaveEndpoint($input: EndpointInput!) {
  saveEndpoint(input: $input) {
    id
    name
    workersMin
    workersMax
  }
}
"""

url = "https://api.runpod.io/graphql"
headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

endpoints = [
    {"id": "ahsqb1dyttja8o", "name": "qwen3-embedding-8b", "workersMin": target, "workersMax": 3, "idleTimeout": 5},
    {"id": "e4vphzghmbvt7j", "name": "Qwen3.5-9B", "workersMin": target, "workersMax": 2, "idleTimeout": 5},
]

for ep in endpoints:
    resp = httpx.post(url, headers=headers, json={
        "query": query,
        "variables": {"input": ep}
    }, timeout=15)
    print(f"Endpoint {ep['id']}:", resp.status_code, resp.json())
