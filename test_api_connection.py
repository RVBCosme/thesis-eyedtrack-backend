#!/usr/bin/env python3
"""
Test script to verify API connectivity from different addresses
"""
import requests
import json

def test_api_endpoint(base_url):
    """Test the /api/latest_behavior endpoint"""
    try:
        url = f"{base_url}/api/latest_behavior"
        print(f"Testing: {url}")
        
        response = requests.get(url, timeout=5)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectTimeout:
        print("Connection timed out")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    # Test different base URLs
    test_urls = [
        "http://localhost:5000",
        "http://127.0.0.1:5000", 
        "http://192.168.68.119:5000",
        "http://10.0.2.2:5000"
    ]
    
    print("Testing API connectivity...")
    print("=" * 50)
    
    for url in test_urls:
        print(f"\n--- Testing {url} ---")
        success = test_api_endpoint(url)
        print(f"Result: {'✓ SUCCESS' if success else '✗ FAILED'}")
        print("-" * 30) 