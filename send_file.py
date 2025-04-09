import requests

# dummy test code
# uninstall requests
session = requests.Session()
response = session.post(
    "http://localhost:8000/detect",
    data={"api_key": "value"},
    files={"image": open("test-image.jpg", "rb")},
)
print(response.text)

# yes its so dummy it needs to be removed