import json
import time

try:
    import requests
except Exception:
    requests = None
    from urllib import request as urllib_request
    import urllib.parse

URL = "http://127.0.0.1:3001/predict/transport"

payloads = [
    # Train: expect lookup-based response
    {
        "transport_type": "train",
        "distance_km": 500,
        "city": "Bengaluru",
        "passengers": 1,
        "train_class": "ordinary passenger"
    },
    # Bus: ML model
    {
        "transport_type": "bus",
        "distance_km": 450,
        "city": "Mumbai",
        "passengers": 1,
        "train_class": None
    },
    # Flight: ML model
    {
        "transport_type": "flight",
        "distance_km": 1200,
        "city": "Delhi",
        "passengers": 1,
        "train_class": None
    }
]

headers = {"Content-Type": "application/json"}

results = []
for p in payloads:
    print(f"\n--- Request: {p['transport_type']} ---")
    data = json.dumps(p).encode('utf-8')
    if requests:
        try:
            r = requests.post(URL, json=p, timeout=10)
            print(f"Status: {r.status_code}")
            try:
                print(r.json())
            except Exception:
                print(r.text)
            results.append((p['transport_type'], r.status_code, r.text))
        except Exception as e:
            print("Request failed:", e)
            results.append((p['transport_type'], 'error', str(e)))
    else:
        try:
            req = urllib_request.Request(URL, data=data, headers=headers)
            with urllib_request.urlopen(req, timeout=10) as resp:
                body = resp.read().decode('utf-8')
                print("Status:", resp.status)
                print(body)
                results.append((p['transport_type'], resp.status, body))
        except Exception as e:
            print("Request failed:", e)
            results.append((p['transport_type'], 'error', str(e)))

print('\nAll tests finished')
for r in results:
    print(r)
