import json
import os

with open("projects/poc/city_district.json") as f:
    DICT_CITY = json.load(f)

LIST_CITY = list(DICT_CITY.keys())