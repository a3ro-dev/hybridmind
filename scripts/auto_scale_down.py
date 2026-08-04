import urllib.request
import json
import time
import os
import datetime

API_KEY = os.environ["RUNPOD_API_KEY"]
GRAPHQL_URL = f"https://api.runpod.io/graphql?api_key={API_KEY}"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Content-Type": "application/json"
}

def scale_down_endpoints():
    # Query endpoints first
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
    try:
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
            with urllib.request.urlopen(req_mut) as resp_mut:
                res = json.loads(resp_mut.read().decode('utf-8'))
                print(f"Successfully scaled down workersMin=0 for {ep['name']} ({ep_id}):", res)
    except Exception as e:
        print("Error during scale down:", e)

def is_memorybench_running():
    res = os.popen("ps aux | grep -E 'index.ts|tsx' | grep -v grep").read()
    return len(res.strip()) > 0

print("=== Starting Auto Scale Down Monitor ===")
print("Will set RunPod minWorkers to 0 when memorybench finishes!")

while True:
    running = is_memorybench_running()
    if not running:
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Benchmark finished! Scaling down...")
        scale_down_endpoints()
        break
    time.sleep(30)
