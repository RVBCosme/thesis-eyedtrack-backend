import requests

def get_video_deets(video_id):
    # Using path parameters
    url = f'http://127.0.0.1:5000/vid/{video_id}'  # Hardcoding the ID into the URL path
    response = requests.get(url)
    # Print the response
    print("Status Code:", response.status_code)
    print("Response Data:", response.json())

def post_app_deets(algo_ear, algo_mar, algo_pitch, algo_roll, algo_yaw, behavior_category, behavior_confidence, behavior_output, timestamp, video_id):
    url = 'http://127.0.0.1:5000/create_app'
    headers = {'Content-type': 'application/json'}
    myobj = {"algo_ear": algo_ear,
             "algo_mar": algo_mar,
             "algo_pitch": algo_pitch,
             "algo_roll": algo_roll,
             "algo_yaw": algo_yaw,
             "behavior_category": behavior_category,
             "behavior_confidence": behavior_confidence,
             "behavior_output": behavior_output,
             "timestamp": timestamp,
             "video_id": video_id}
    
    x = requests.post(url, json = myobj, headers=headers)
    print(x.text)

# post_app_deets(algo_ear=0.5, 
#                algo_mar=0.5, 
#                algo_pitch=0.5, 
#                algo_roll=0.5, 
#                algo_yaw=0.5, 
#                behavior_category="category", 
#                behavior_confidence="confidence", 
#                behavior_output="output", 
#                timestamp="2025-05-27 01:01:01", 
#                video_id=6)
get_video_deets(video_id=6)



