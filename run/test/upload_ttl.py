import requests


def upload_ttl_to_fuseki(file_path, fuseki_url, username, password):
    headers = {"Content-Type": "text/turtle;charset=utf-8"}

    with open(file_path, "r") as file:
        data = file.read()

    response = requests.post(
        f"{fuseki_url}/data", headers=headers, data=data, auth=(username, password)
    )

    if response.status_code == 200:
        print("Data uploaded successfully.")
    else:
        print(f"Failed to upload data. Status code: {response.status_code}")
        print(f"Response: {response.text}")


ttl_file_path = "data/onto/criminal.v2.ttl"
fuseki_endpoint = "http://localhost:3032/legal"
username = "admin"
password = "abc123-qwe"

upload_ttl_to_fuseki(ttl_file_path, fuseki_endpoint, username, password)
