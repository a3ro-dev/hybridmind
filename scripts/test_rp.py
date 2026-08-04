import urllib.request
import json
import urllib.error
import os

API_KEY = os.environ["RUNPOD_API_KEY"]
GRAPHQL_URL = f"https://api.runpod.io/graphql?api_key={API_KEY}"

query = """
query {
  myself {
    endpoints {
      id
      name
      templateId
      gpuIds
      idleTimeout
      scalerType
      scalerValue
      workersMin
      workersMax
      gpuCount
    }
  }
}
"""

req = urllib.request.Request(
    GRAPHQL_URL,
    data=json.dumps({"query": query}).encode('utf-8'),
    headers={"Content-Type": "application/json"}
)
try:
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        print("Query success:", data)
except urllib.error.HTTPError as e:
    print("HTTP Error:", e.code, e.read().decode('utf-8'))
