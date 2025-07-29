import requests, yaml

with open("telegram.yaml") as f:
    config = yaml.safe_load(f)["telegram"]

url = f"https://api.telegram.org/bot{config['token']}/sendMessage"
data = {
    "chat_id": config["chat_id"],
    "text": "✅ Test message from WindowSurfer bot"
}

r = requests.post(url, data=data)
print("Status:", r.status_code)
print("Response:", r.json())