import requests

# dummy test code
# uninstall requests
host = "http://localhost:8080"
session = requests.Session()
image = "D:\\AI\\false_positives\\2025-07-10T03-49-50.571_person_deer.jpg"
response = session.post(
    f"{host}/detect",
    data={"api_key": "value"},
    files={"image": open(image, "rb")},
)
print("old")
print(response.text)

response = session.post(
    f"{host}/detect",
    data={"api_key": "value"},
    files={"image": open(image, "rb")},
)
print("new")
print(response.text)
