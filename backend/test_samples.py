import requests
import json

# Sample 1: Mumbai to Goa - Bus Trip
sample1 = {
    "startLocation": "Mumbai",
    "destination": "Goa",
    "country": "India",
    "transportType": "Bus",
    "busType": "Volvo",
    "numberOfDays": 5,
    "accommodationType": "3Star",
    "foodPreference": "Medium",
    "activityPreference": "High",
    "distanceKm": 450
}

# Sample 2: Delhi to Jaipur - Train Trip
sample2 = {
    "startLocation": "Delhi",
    "destination": "Jaipur",
    "country": "India",
    "transportType": "Train",
    "trainType": "AC_2A",
    "numberOfDays": 3,
    "accommodationType": "4Star",
    "foodPreference": "High",
    "activityPreference": "Medium",
    "distanceKm": 280
}

print("SAMPLE INPUT 1: Mumbai to Goa (Bus)")
print(json.dumps(sample1, indent=2))
response1 = requests.post("http://localhost:8000/api/estimate", json=sample1)
print("\nOUTPUT 1:")
print(json.dumps(response1.json(), indent=2))

print("\n" + "="*70 + "\n")

print("SAMPLE INPUT 2: Delhi to Jaipur (Train)")
print(json.dumps(sample2, indent=2))
response2 = requests.post("http://localhost:8000/api/estimate", json=sample2)
print("\nOUTPUT 2:")
print(json.dumps(response2.json(), indent=2))
