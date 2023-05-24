import osmnx as ox
import requests
import json
import pandas as pd
import difflib

def extract_address(url, address):
    # url = "http://localhost:8080/predictions/ner_model"
    payload = json.dumps({
  "instances": 
    [{
      "address": address
    }]
})
    headers = {
    'Content-Type': 'application/json'
    }
    response = requests.request("POST", url+"/predictions/ner_model", headers=headers, data=payload)
    response = response.json()
    data = response['predictions'][0]
    return data

def get_score(data):
    score = 0
    if data.get("road_address", "") != "":
        score += 41
    if data.get("number_address", "") != "":
        score += 31
    if data.get("rt_address", "") != "":
        score += 11
    if data.get("rw_address", "") != "":
        score += 11
    if data.get("block_address", "") != "":
        score += 6
    return score

def calculate_similarity(x,y):
    x = x.lower()
    y = y.lower()
    return difflib.SequenceMatcher(None, x, y).quick_ratio()

def construct_address(data, district, city):
    blok = f"blok {data['block_address']}" if data["block_address"] != "" else ""
    nomor = f"no {data['number_address']}" if data["number_address"] != "" else ""
    address = f"jalan {data['road_address']} {blok} {nomor} {district} {city}"
    return address.strip()

def get_status_address(api_key, address, poi, radius, extract_address):
    address = address.lower()
    try:
        url = f"https://geocode.search.hereapi.com/v1/geocode?q={'+'.join(address.split())}&apiKey={api_key}"
        response = requests.request("GET", url, headers={}, data={})
        geo = json.loads(response.text)
        if len(geo['items']) == 0:
            return {"label": address, "postalCode": ""}, "unknown", "unknown", "unknown", "unknown", "unknown", "unknown", "unknown", "unknown"
        item = geo['items'][0]
        district_score = calculate_similarity(extract_address['district'].lower(), item['address']['district'].lower()) * 100
        city_score = calculate_similarity(extract_address['city'].lower(), item['address']['city'].lower()) * 100
        road_score = 100
        if extract_address['road_address'] != "":
            road_score = calculate_similarity(f"jalan {extract_address['road_address']}".lower() , item['address']['street'].lower()) * 100
        if extract_address['number_address'] != "":
            number_score = calculate_similarity(extract_address['number_address'], item['address']['houseNumber']) * 100
            confidence_score = 0.4*((district_score+city_score)/2) + 0.6*((road_score+number_score)/2)
        else:
            confidence_score = 0.4*((district_score+city_score)/2) + 0.6*(road_score)
        
        lat_lon = item['position']
        if item['resultType'] == 'place':
            categories = item['categories']
            categories = ",".join([x['name'].lower() for x in categories])
        else:
            categories = "street"
        address_ = item['address']
        lat_lon['lat'] = lat_lon['lat']
        lat_lon['lng'] = lat_lon['lng']
        lat_lon = [y for x,y in lat_lon.items()]
        lat, long = lat_lon
        # url = f"https://discover.search.hereapi.com/v1/discover?q={'+'.join(poi.split())}&in=circle:{lat},{long};r={radius}&apiKey={api_key}"
        # response_poi = requests.request("GET", url, headers={}, data={})
        # discover = json.loads(response_poi.text)
        # items_poi = discover['items']
        # num_of_poi = len(items_poi)
        # min_distance_to_poi = 0
        # max_distance_to_poi = 0
        # if num_of_poi > 0:
        #     min_distance_to_poi = items_poi[0]['distance']
        #     max_distance_to_poi = items_poi[-1]['distance']
        # g = geocoder.osm(lat_lon, method='reverse', maxRows=10)
        # type_address = 'office' if g.current_result.type in ["office", "commercial", "company"] else "not office"
        h = ox.geometries_from_point(lat_lon,tags={'highway': True}, dist=100)
        max_lanes = 'unknown'
        max_width = 'unknown'
        is_motorcycle = 'unknown'
        surface = 'unknown'
        if "name" in h.columns and "way" in h.index:
            way = h.loc['way']
            way = way.loc[(~way.name.isna())]
            way = way.loc[~way.name.str.lower().str.contains("tol")]
            way['score'] = way.name.apply(calculate_similarity, y=item['address']['street'])
            way = way.loc[way.score > 0.8]
            if "lanes" in way.columns:
                way = way.loc[(~way.lanes.isna())]
                way['lanes'] = way.lanes.astype(float)
                max_lanes = way.lanes.max()
            else:
                first_row = way.iloc[0]

            if "width" in way.columns:
                way['width'] = way.width.astype(float)
                max_width = way.width.max()
                first_row = way.loc[way.width == max_width].iloc[0]
            elif 'est_width' in way.columns:
                way['est_width'] = way.est_width.astype(float)
                max_width = way.est_width.max()
                first_row = way.loc[way.est_width == max_width].iloc[0]
            else:
                first_row = way.iloc[0]
            # first_row = first_row.loc[~first_row.name.str.lower().str.contains("tol")].iloc[0]
            is_motorcycle = first_row.motorcycle if "motorcycle" in way.columns else "unknown"
            is_motorcycle = "unknown" if pd.isna(is_motorcycle) else is_motorcycle
            surface = first_row.surface if "surface" in way.columns else "unknown"
            surface = "unknown" if pd.isna(surface) else surface
            num_of_poi = None
            min_distance_to_poi = None
            max_distance_to_poi = None
        return address_, lat_lon[0], lat_lon[1], num_of_poi, categories, max_lanes, max_width, is_motorcycle, surface, confidence_score, min_distance_to_poi, max_distance_to_poi
    except Exception as e:
        return {"label": address, "postalCode": ""}, "error", "error", "error", "error", "error", "error", "error", "error", 0, 0, 0