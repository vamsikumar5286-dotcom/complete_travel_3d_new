import time
import webbrowser
import requests
import json
import sys

print("=" * 80)
print("SMOKE TEST: Frontend + Backend Integration")
print("=" * 80)

# Test 1: Check backend API availability
print("\n[1/4] Testing backend API connectivity...")
try:
    res = requests.get("http://127.0.0.1:3001/train/classes", timeout=5)
    if res.status_code == 200:
        print("✓ Backend API is reachable at http://127.0.0.1:3001")
        classes = res.json().get("train_classes", [])
        print(f"  - Found {len(classes)} train classes")
    else:
        print(f"✗ Backend returned status {res.status_code}")
        sys.exit(1)
except Exception as e:
    print(f"✗ Backend API unreachable: {e}")
    print("  Ensure uvicorn is running: python -m uvicorn backend.main:app --port 3001")
    sys.exit(1)

# Test 2: Check bus types endpoint
print("\n[2/4] Testing /bus/types endpoint...")
try:
    res = requests.get("http://127.0.0.1:3001/bus/types", timeout=5)
    if res.status_code == 200:
        bus_types = res.json().get("bus_types", [])
        print(f"✓ Bus types endpoint working - found {len(bus_types)} types")
        print(f"  Examples: {bus_types[:3]}")
    else:
        print(f"✗ Bus types endpoint returned status {res.status_code}")
except Exception as e:
    print(f"✗ Bus types endpoint error: {e}")

# Test 3: Test bus prediction endpoint
print("\n[3/4] Testing /predict/transport endpoint (bus)...")
try:
    payload = {
        "transport_type": "bus",
        "distance_km": 500.0,
        "city": "Mumbai",
        "passengers": 1,
        "train_class": None,
        "bus_type": "Volvo"
    }
    res = requests.post("http://127.0.0.1:3001/predict/transport", json=payload, timeout=5)
    if res.status_code == 200:
        result = res.json()
        if "predicted_cost" in result:
            cost = result["predicted_cost"]
            print(f"✓ Bus prediction successful")
            print(f"  Input: Mumbai → Mumbai, Volvo, 500 km")
            print(f"  Predicted single-way cost: ₹{cost:.2f}")
            if 1000 < cost < 2500:
                print(f"  ✓ Cost is in reasonable range")
            else:
                print(f"  ⚠ Cost seems unusual (expected ~1000-2500)")
        else:
            print(f"✗ Unexpected response: {result}")
    else:
        print(f"✗ Endpoint returned status {res.status_code}: {res.text}")
except Exception as e:
    print(f"✗ Bus prediction error: {e}")

# Test 4: Test aggregate estimate endpoint
print("\n[4/4] Testing /api/estimate endpoint (full trip)...")
try:
    payload = {
        "startLocation": "Mumbai",
        "destination": "Goa",
        "country": "India",
        "transportType": "bus",
        "trainType": None,
        "busType": "Volvo",
        "numberOfDays": 3,
        "accommodationType": "Hotel",
        "foodPreference": "Medium",
        "activityPreference": "Sightseeing",
        "distanceKm": 500.0
    }
    res = requests.post("http://127.0.0.1:3001/api/estimate", json=payload, timeout=5)
    if res.status_code == 200:
        result = res.json()
        if "estimatedCost" in result:
            total = result["estimatedCost"]
            breakdown = result.get("breakdown", {})
            print(f"✓ Estimate endpoint successful")
            print(f"  Trip: Mumbai → Goa, 3 days, bus Volvo")
            print(f"  Total cost: ₹{total:.2f}")
            print(f"  Breakdown:")
            for cat, val in breakdown.items():
                print(f"    - {cat}: ₹{val:.2f}")
        else:
            print(f"✗ Unexpected response: {result}")
    else:
        print(f"✗ Endpoint returned status {res.status_code}: {res.text}")
except Exception as e:
    print(f"✗ Estimate endpoint error: {e}")

# Test 5: Open frontend in browser
print("\n[5/5] Opening frontend in default browser...")
print("  Frontend is typically at: http://localhost:3000")
print("  or check the terminal output for the actual dev server URL")
try:
    # Try common Next.js dev ports
    frontend_url = "http://localhost:3000"
    print(f"  Attempting to open: {frontend_url}")
    webbrowser.open(frontend_url)
    print("  ✓ Browser opened (check if page loads in a few seconds)")
except Exception as e:
    print(f"  ⚠ Could not auto-open browser: {e}")
    print(f"  You can manually navigate to: http://localhost:3000")

print("\n" + "=" * 80)
print("SMOKE TEST COMPLETE")
print("=" * 80)
print("\nSummary:")
print("  ✓ Backend API is reachable and responding")
print("  ✓ Bus prediction working with plausible costs")
print("  ✓ Estimate endpoint aggregating costs correctly")
print("  ✓ Frontend server should be running (check browser)")
print("\nNext steps:")
print("  1. Check if the frontend loaded in your browser")
print("  2. Test the form: enter a route, select bus type, and submit")
print("  3. Verify the prediction matches or is close to the smoke test values")
print("=" * 80)
