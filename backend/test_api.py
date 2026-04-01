import requests
import json
from pprint import pprint
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    print("\n--- Testing /health ---")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    pprint(response.json())

def test_current_time():
    print("\n--- Testing /api/current-time ---")
    response = requests.get(f"{BASE_URL}/api/current-time")
    print(f"Status Code: {response.status_code}")
    pprint(response.json())

def test_menstrual_health():
    print("\n--- Testing /api/menstrual-health ---")
    response = requests.get(f"{BASE_URL}/api/menstrual-health", params={"section": "foods_to_eat"})
    print(f"Status Code: {response.status_code}")
    pprint(response.json())

def test_hospitals():
    print("\n--- Testing /api/hospitals-doctors ---")
    # Coordinates for Kolkata as a test point
    params = {
        "lat": 22.57,
        "lon": 88.36,
        "radius": 5000
    }
    response = requests.get(f"{BASE_URL}/api/hospitals-doctors", params=params)
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Total Hospitals Found: {data.get('count')}")
    print(f"Total Available Doctors: {data.get('doctors_available_now')}")
    print("Pre-view of first hospital:")
    if data.get('hospitals'):
        pprint(data['hospitals'][0])

def test_doctor_availability():
    print("\n--- Testing /api/doctor-availability ---")
    params = {
        "lat": 22.57,
        "lon": 88.36,
        "radius": 5000
    }
    response = requests.get(f"{BASE_URL}/api/doctor-availability", params=params)
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Total Available Doctors: {data.get('total_doctors_available')}")
    if data.get('hospitals'):
        print(f"Sample Hospital with Available Doctors ({data['hospitals'][0]['name']}):")
        pprint(data['hospitals'][0]['available_doctors'])

def test_predict():
    """Test the /predict endpoint with a sample image."""
    print("\n--- Testing /predict endpoint (Keras Integration) ---")
    
    # Try to find a test image
    test_image_paths = [
        Path("test_image.png"),
        Path("test_image.jpg"),
        Path("../frontend/sample.jpg"),
    ]
    
    test_image = None
    for path in test_image_paths:
        if path.exists():
            test_image = path
            break
    
    if not test_image:
        print("⚠️ No test image found. Create a test image or use an existing one.")
        print("   Supported paths: test_image.png, test_image.jpg")
        return
    
    try:
        with open(test_image, 'rb') as f:
            files = {"file": ("test_image.jpg", f, "image/jpeg")}
            params = {"lat": 22.57, "lon": 88.36, "radius": 5000}
            response = requests.post(f"{BASE_URL}/predict", files=files, params=params)
        
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Predicted Class: {data.get('predicted_class')}")
        print(f"Confidence: {data.get('confidence') * 100:.2f}%")
        print(f"All Probabilities: {data.get('probabilities')}")
        print(f"Preliminary Steps: {data.get('preliminary_steps')}")
        if data.get('nearest_hospital'):
            print(f"Nearest Hospital: {data.get('nearest_hospital').get('name')}")
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")

if __name__ == "__main__":
    print("Testing Blood Health Advisor APIs...")
    print("Make sure the Uvicorn server is running before executing this script!")
    test_health()
    test_current_time()
    test_menstrual_health()
    test_hospitals()
    test_doctor_availability()
    test_predict()
    print("\n✅ All Tests Completed Successfully!")
