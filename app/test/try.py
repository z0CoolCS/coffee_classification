import requests

url = ""


resp = requests.post(url, files={'file': open("cafe_class1.jpg", "rb")}, timeout=30)
#resp = requests.get(url)

print(resp.json())

