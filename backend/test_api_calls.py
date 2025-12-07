import os
import sys
import json
from importlib import util

# Ensure backend directory is on sys.path so imports like 'train_utils' resolve
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Load main.py as a module named 'main'
main_path = os.path.join(BASE_DIR, 'main.py')
spec = util.spec_from_file_location('main', main_path)
main = util.module_from_spec(spec)
spec.loader.exec_module(main)

from fastapi.testclient import TestClient
# Explicitly call startup loader to ensure models are loaded when using TestClient
try:
    main.load_models()
except Exception as e:
    print('Warning: main.load_models() raised:', e)

client = TestClient(main.app)

cases = []

# 1) Bus
cases.append(('Bus (Mumbai->Goa, Volvo, 500km)', '/predict/transport', {
    'transport_type': 'bus',
    'distance_km': 500.0,
    'city': 'Mumbai',
    'passengers': 1,
    'train_class': None,
    'bus_type': 'Volvo'
}))

# 2) Train
cases.append(('Train (Mumbai->Goa, AC_3A, 500km)', '/predict/transport', {
    'transport_type': 'train',
    'distance_km': 500.0,
    'city': 'Mumbai',
    'passengers': 1,
    'train_class': 'AC_3A',
    'bus_type': None
}))

# 3) API Estimate (aggregate)
cases.append(('Estimate (roundtrip Mumbai->Goa, bus Volvo)', '/api/estimate', {
    'startLocation': 'Mumbai',
    'destination': 'Goa',
    'country': 'India',
    'transportType': 'bus',
    'trainType': None,
    'busType': 'Volvo',
    'numberOfDays': 3,
    'accommodationType': 'Hotel',
    'foodPreference': 'Medium',
    'activityPreference': 'Sightseeing',
    'distanceKm': 500.0
}))

print('Running API tests...')
for title, path, payload in cases:
    print('\n---')
    print(title)
    print('POST', path)
    print('Payload:', json.dumps(payload))
    try:
        res = client.post(path, json=payload)
        print('Status code:', res.status_code)
        try:
            print('Response JSON:', json.dumps(res.json(), indent=2))
        except Exception:
            print('Response text:', res.text)
    except Exception as e:
        print('Request failed:', e)

print('\nAll tests completed.')
