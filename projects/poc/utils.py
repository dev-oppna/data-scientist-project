import requests
import json

base_url = "https://api-stg.oppna.dev"

def get_aggregated(name: str, nik: str, phone: str, address: str, email: str, city: str, district: str) -> dict:
    headers = {
        "Content-Type": "application/json"
    }
    data = json.dumps({
        "name": name,
        "nik": nik,
        "phone": phone,
        "address": address,
        "email": email,
        "city": city,
        "district": district
    })
    ses = requests.session()
    try:
        resp = ses.post(f"{base_url}/v1/guardians/aggregated", headers=headers, data=data)
        response = resp.json()
        return response
    except Exception as e:
        return {"data": None}

def get_detailed(name: str, nik: str, phone: str, address: str, email: str) -> dict:
    headers = {
        "Content-Type": "application/json"
    }
    data = json.dumps({
        "name": name,
        "nik": nik,
        "phone": phone,
        "address": address,
        "email": email,
        "city": "abc",
        "district": "bcsa"
    })
    ses = requests.session()
    try:
        resp = ses.post(f"{base_url}/v1/guardians/detailed", headers=headers, data=data)
        response = resp.json()
        return response
    except Exception as e:
        return {}
    
def get_detailed_retrieve_phone(name: str, phone: str) -> dict:
    headers = {
        "Content-Type": "application/json"
    }
    data = json.dumps({
        "name": name,
        "nik": "-",
        "phone": phone,
        "address": "-",
        "email": "-",
        "city": "abc",
        "district": "bcsa"
    })
    ses = requests.session()
    try:
        resp = ses.post(f"{base_url}/v1/guardians/retrieve", headers=headers, data=data, timeout=5)
        response = resp.json()
        if response["data"]:
            return response["data"]
        else:
            return {"addresses": [], "summary": [], "address": [], "email": [], "nik": [], "pln": []}
    except Exception as e:
        return {"addresses": [], "summary": [], "address": [], "email": [], "nik": [], "pln": []}
    
def get_address_verification_validation(phone: str, address: str, district:str, city:str, poi:str, radius:int) -> dict:
    headers = {
        "Content-Type": "application/json"
    }
    data = json.dumps({
        "phone": phone,
        "address": f"{address}, {district}, {city}",
        "poi": poi,
        "radius": int(radius)
    })
    ses = requests.session()
    try:
        resp = ses.post(f"{base_url}/v1/marlins", headers=headers, data=data)
        response = resp.json()
        return response["data"]
    except Exception as e:
        return {}
    
def get_opas_by_address(address_id: str) -> dict:
    headers = {
        "Content-Type": "application/json"
    }
    data = json.dumps({
        "address_id": address_id
    })
    ses = requests.session()
    try:
        resp = ses.post(f"{base_url}/v1/guardians/address/opas", headers=headers, data=data, timeout=3)
        response = resp.json()
        if response["data"]:
            return response
        else:
            return {"data": {"phones": []}}
    except Exception as e:
        return {"data": {"phones": []}}
    
def masking_static(x: str) -> str:
    if len(x) > 2:
        x = "".join([k if ind < 2 else "*" for ind, k in enumerate(x)])
    return x
    
def masking_name(name: str) -> str:
    name = name.split()
    name = [masking_static(x) for x in name]
    return " ".join(name)

def transform_date(date: str) -> str:
    if date:
        json_month = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
        split_date = date.split("-")
        year = split_date[0]
        month = json_month[int(split_date[1])]
        return f"{month} {year}"
    return date
    
def transform_state_court(data: list) -> list:
    data_list = []
    for dat in data:
        case = dat["case_classification"]
        location = dat["location"]
        role = dat["role"]
        for date, url in zip(dat["registration_date"], dat["url"]):
            data_list.append({"case": case, "location": location, "role": role, "date": date, "url": url})
    return data_list

def transform_addresses(data: list) -> list:
    data = [{k:l for k,l in x.items() if k not in ["building_name_address", "road_address", "block_address", "number_address", "rt_address", "rw_address", "remarks", "meta_loaded_at", "completeness_level", "address_quality_score"]} for x in data]
    return data
        
def get_detailed_link_analysis(opa: str) -> dict:
    headers = {
        "Content-Type": "application/json"
    }
    data = json.dumps({
        "opa_id": opa,
    })
    ses = requests.session()
    try:
        resp = ses.post(f"{base_url}/v1/guardians/relations", headers=headers, data=data, timeout=60)
        response = resp.json()
        if response["data"]:
            return response["data"]
        else:
            return {"nodes": [], "edges": []}
    except Exception as e:
        return {"nodes": [], "edges": []}


def construct_address(data, district, city):
    blok = f"blok {data['block']}" if data["block"] != "" else ""
    remarks = f"{data['remark']}" if data["remark"] != "" else ""
    nomor = f"no {data['number']}" if data["number"] != "" else ""
    rt = f"rt {data['rt']}" if data["rt"] != "" else ""
    rw = f"rw {data['rw']}" if data["rw"] != "" else ""
    if data['road_name'] != "":
        address = f"jalan {data['road_name']} {blok} {nomor} {remarks} {district} {city}"
    else:
        address = f"{blok} {nomor} {rt} {rw} {district} {city}"
    address = " ".join(address.split())
    return address.strip()

def get_formatted(data, district, city, province):  
    return (f"{data['building_name']} " + construct_address(data, district, city) + f" {province}").strip().lower()