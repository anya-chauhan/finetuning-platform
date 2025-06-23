# test_api.py
import requests

# Test if API is running and endpoints are accessible
base_url = "http://localhost:8000"

print("Testing API endpoints...")

# Test root
try:
    response = requests.get(f"{base_url}/")
    print(f"Root endpoint: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error accessing root: {e}")

# Test contexts
try:
    response = requests.get(f"{base_url}/contexts")
    print(f"\nContexts endpoint: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Number of contexts: {data.get('total_contexts', 0)}")
        print(f"First 3 contexts: {data.get('contexts', [])[:3]}")
except Exception as e:
    print(f"Error accessing contexts: {e}")

# Test proteins
try:
    response = requests.get(f"{base_url}/proteins?limit=5")
    print(f"\nProteins endpoint: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Total proteins: {data.get('total_proteins', 0)}")
except Exception as e:
    print(f"Error accessing proteins: {e}")