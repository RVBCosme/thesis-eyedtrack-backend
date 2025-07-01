#!/usr/bin/env python3

import requests
import json

def test_latest_behavior():
    """Test the latest_behavior endpoint"""
    
    try:
        # Test the endpoint
        url = "http://192.168.68.119:5000/api/latest_behavior"
        print(f"Testing endpoint: {url}")
        
        response = requests.get(url, timeout=5)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ SUCCESS!")
            print(f"Response: {json.dumps(data, indent=2)}")
            
            # Check behavior flags
            behavior_category = data.get('behavior_category', {})
            is_drowsy = behavior_category.get('is_drowsy', False)
            is_yawning = behavior_category.get('is_yawning', False)
            is_distracted = behavior_category.get('is_distracted', False)
            
            print(f"\n🚨 Behavior Flags:")
            print(f"  - Drowsy: {is_drowsy}")
            print(f"  - Yawning: {is_yawning}")
            print(f"  - Distracted: {is_distracted}")
            
            if is_drowsy or is_yawning or is_distracted:
                print("⚠️ RISKY BEHAVIOR DETECTED!")
            else:
                print("✅ Normal behavior")
                
        else:
            print(f"❌ ERROR: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR: Could not connect to server")
        print("Make sure the backend server is running on http://192.168.68.119:5000")
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    test_latest_behavior() 