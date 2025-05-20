import json
import requests
import os

base_url = "https://api-stg.oppna.dev"
basic_panthers = os.getenv("basic_panthers")
token_panthers = os.getenv("token_panthers")

def get_watchlist(name: str, phone: str, nik: str, email:str, dob:str, location:str) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Basic {basic_panthers}",
        "Token": {token_panthers}
    }
    data = json.dumps({
        "name" : name,
        "phone": phone,
        "nik": nik,
        "email": email,
        "dob": dob,
        "location": location
    })
    
    ses = requests.session()
    try:
        resp = ses.post(f"{base_url}/v1/panthers", headers=headers, data=data)
        response = resp.json()
        return response
    except Exception as e:
        raise e
        return {}
