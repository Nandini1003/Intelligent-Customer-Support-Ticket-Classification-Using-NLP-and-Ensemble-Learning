import requests

# Flask API URL
url = "http://127.0.0.1:5000/predict"

# Example ticket messages
tickets = [
    "I want a refund for my invoice",
    "Cannot login to my account",
    "Delivery is delayed"
]

for msg in tickets:
    response = requests.post(url, json={"message": msg})
    if response.status_code == 200:
        print(response.json())
    else:
        print("Error:", response.status_code, response.text)
