import json
import requests

base_url = "https://api-stg.oppna.dev/v1/cardinals/opa/"

def get_enrichment(opa_id: str, name: str) -> dict:
    headers = {
        "Content-Type": "application/json"
    }
    data = json.dumps({
        "date": "latest",
        "name": name
    })
    ses = requests.session()
    try:
        resp = ses.get(f"{base_url}{opa_id}", headers=headers, data=data)
        response = resp.json()
        return response
    except Exception as e:
        return {}