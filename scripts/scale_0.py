import urllib.request
import json
import os

API_KEY = os.environ["RUNPOD_API_KEY"]
GRAPHQL_URL = f"https://api.runpod.io/graphql?api_key={API_KEY}"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Content-Type": "application/json"
}

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

req = urllib.request.Request(GRAPHQL_URL, data=json.dumps({"query": query}).encode('utf-8'), headers=HEADERS)
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode('utf-8'))
    endpoints = data["data"]["myself"]["endpoints"]

for ep in endpoints:
    ep_id = ep["id"]
    mutation = f"""
    mutation {{
        saveEndpoint(input: {{
            id: "{ep_id}",
            name: "{ep['name']}",
            templateId: "{ep['templateId']}",
            gpuIds: "{ep['gpuIds']}",
            idleTimeout: {ep['idleTimeout']},
            scalerType: "{ep['scalerType']}",
            scalerValue: {ep['scalerValue']},
            workersMin: 0,
            workersMax: {ep['workersMax']},
            gpuCount: {ep['gpuCount']}
        }}) {{
            id
            name
            workersMin
        }}
    }}
    """
    req_mut = urllib.request.Request(GRAPHQL_URL, data=json.dumps({"query": mutation}).encode('utf-8'), headers=HEADERS)
    with urllib.request.urlopen(req_mut) as resp_m:
        print(f"Scaled down {ep['name']} ({ep_id}) to 0:", json.loads(resp_m.read().decode('utf-8')))
