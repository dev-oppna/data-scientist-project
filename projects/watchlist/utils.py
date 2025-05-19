import json
import requests

base_url = "https://api-stg.oppna.dev"

def get_watchlist(name: str, phone: str, nik: str, email:str, dob:str, location:str) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Basic b3BwbmE6b3BwbmFwYXNzd29yZDEyMw==",
        "Token": "6f9e4ebb7e868aa084788dd576337057afef823d3402ba5fd2fb135b6947c29d"
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
