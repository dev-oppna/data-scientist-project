import numpy as np
import networkx as nx
from haversine import haversine
import itertools
from scipy.spatial import distance
import networkx.algorithms.approximation as nx_app
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import string
import requests
import re
import collections
import json
import jaro
from statistics import mean

def extract_address(address, url):
    payload = json.dumps({
        "instances": 
            [{
            "address": address
            }]
        })
    headers = { 'Content-Type': 'application/json' }
    response = requests.request("POST", url+"/predictions/ner_model", headers=headers, data=payload)
    response = response.json()
    data = response['predictions'][0]
    return data

def extract_address_bulk(df, url):
    df1 = df.copy()
    df1 = df1.loc[:,["recipient_address", "district", "city"]]
    df1.drop_duplicates(inplace=True)
    payload = json.dumps({
        "instances": 
            [{
                "address": row[1]["recipient_address"],
                "district": row[1]["district"],
                "city": row[1]["city"],
                "list_address": []
            } for row in df1.iterrows()]
        })
    headers = { 'Content-Type': 'application/json' }
    response = requests.request("POST", url+"/predictions/giants", headers=headers, data=payload)
    response = response.json()
    addresses = [f"{row[1]['recipient_address']} {row[1]['district']} {row[1]['city']}" for row in df1.iterrows()]
    data = response['predictions']
    data = {y:dat["extracted_address"] for dat, y in zip(data,addresses)}
    return data

DICT_FIX = {
    "jl": "jalan",
    "jln": "jalan",
    "jlan": "jalan",
    "jaln": "jalan",
    "dket": "dekat",
    "dkt": "dekat",
    "dkat": "dekat",
    "deket": "dekat",
    "dekt": "dekat",
    "rmh": "rumah",
    "rmah": "rumah",
    "rumh": "rumah",
    "rm": "rumah makan",
    "rs": "rumah sakit",
    "dpn": "depan",
    "dpan": "depan",
    "depn": "depan",
    "blakng": "belakang",
    "blkg": "belakang",
    "kec": "kecamatan",
    "kab": "kabupaten",
    "ds": "desa",
    "kel": "kelurahan",
    "nomor": "no",
    "gg": "gang",
    "kp": "kampung",
    "kav": "kavling",
    "perum": "perumahan",
    "ged": "gedung",
    "lap": "lapangan", 
    "apartment": "apartement",
    "apt": "apartement",
    "apartemen": "apartement",
    "apartmen": "apartement",
    "ressidence": "residence",
    "komp": "komplek",
    "ds": "desa",
    "dk": "desa",
    "kp": "kampung",
    "ponpes": "pondok pesantren",
    "jend": "jendral",
    "kom": "komplek",
    "ps": "pasar",
    "psr": "pasar"
}

def clean_address(address):
    if isinstance(address, str):
        address = address.lower()
        address = address.replace("alamat", "")
        address = address.replace(".", " ")
        address = address.replace(",", " ")
        address = address.replace("/", " ")
        address = address.replace("-", " ")
        address = address.replace("\n", " ")
        address = re.sub('\(\s*(.*?)\s*\)', r'(\1)', address)
        address = re.sub(r"(\S)\(", r'\1 (', address)
        address = re.sub(r"\)(\S)", r') \1', address)
        address = re.sub('[^a-zA-Z0-9 \n\.]', ' ', address)
        address = re.sub(' +', ' ', address)
        address = re.sub(r' rt([0-9]+)', r" rt \1", address)
        address = re.sub(r' rw([0-9]+)', r" rw \1", address)
        address = re.split(r'(\d+)', address)
        address = [x.strip() for x in address]
        address = " ".join(address)
        address = address.strip()
        address = address.split(" ")
        address = [DICT_FIX.get(x,x) for x in address]
        address = " ".join(address)
        address = re.sub(' +', ' ', address)
        return address
    else:
        return address
    
def covert_to_roman(num):
    # Storing roman values of digits from 0-9
    # when placed at different places
    num = int(num)
    m = ["", "M", "MM", "MMM"]
    c = ["", "C", "CC", "CCC", "CD", "D",
         "DC", "DCC", "DCCC", "CM "]
    x = ["", "X", "XX", "XXX", "XL", "L",
         "LX", "LXX", "LXXX", "XC"]
    i = ["", "I", "II", "III", "IV", "V",
         "VI", "VII", "VIII", "IX"]
    # Converting to roman
    thousands = m[num // 1000]
    hundreds = c[(num % 1000) // 100]
    tens = x[(num % 100) // 10]
    ones = i[num % 10]
    ans = (thousands + hundreds +
           tens + ones)
    return ans

def value(r):
    if (r == 'I'):
        return 1
    if (r == 'V'):
        return 5
    if (r == 'X'):
        return 10
    if (r == 'L'):
        return 50
    if (r == 'C'):
        return 100
    if (r == 'D'):
        return 500
    if (r == 'M'):
        return 1000
    return -1
 
def roman_to_int(string):
    res = 0
    i = 0
    while (i < len(string)):
        # Getting value of symbol s[i]
        s1 = value(string[i])
        if (i + 1 < len(string)):
            # Getting value of symbol s[i + 1]
            s2 = value(string[i + 1])
            # Comparing both values
            if (s1 >= s2):
                # Value of current symbol is greater
                # or equal to the next symbol
                res = res + s1
                i = i + 1
            else:
                # Value of current symbol is greater
                # or equal to the next symbol
                res = res + s2 - s1
                i = i + 2
        else:
            res = res + s1
            i = i + 1
    return str(res)

def validate_roman(string):
    string = string.upper()
    # Importing regular expression
    # Searching the input string in expression and
    # returning the boolean value
    return bool(re.search(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$",string))

def standarized_address(address):
    address = clean_address(address)
    mapping = ['masjid', "banjar", "dusun", "ruko", "taman", "blok", "block", "no", "nomor", 
                "jalan", "perumahan", "al", "city", "residence", "raya", "the", "kavling",
               "kampung", "lapangan", "apartement", "tower", "graha", " kos ", "plaza", "komplek", 
               "gang", "gedung", "lantai", "desa", "pondok", "pesantren", "kantor"]
    address = address.lower()
    address = " ".join([x for x in address.split() if x not in mapping])
#     for key in mapping:
#         address = address.replace(key, ' ')
    if any([validate_roman(x) for x in address.split()]):
        road_address = " ".join([ roman_to_int(x) if validate_roman(x) else x for x in address.split()])
        address = road_address
    address = "".join([x for x in address if not x.isnumeric()])
    address = " ".join([x for x in address.split() if len(x) > 1])
    address = address.translate(str.maketrans('', '', string.punctuation))
    return address.lower().strip()

def standarized_building(building):
    address = clean_address(building)
    mapping = ["gedung", "lantai", "indonesia", "pt", "unit", "apartement"]
    address = address.lower()
    address = " ".join([x for x in address.split() if x not in mapping])
    if any([validate_roman(x) for x in address.split()]):
        road_address = " ".join([ roman_to_int(x) if validate_roman(x) else x for x in address.split()])
        address = road_address
    address = "".join([x for x in address if not x.isnumeric()])
    address = " ".join([x for x in address.split() if len(x) > 1])
    address = address.translate(str.maketrans('', '', string.punctuation))
    return address.lower().strip()

def standarized_cluster_name(address, type_):
    address_ = standarized_address(address)
    if len(address) > 15 and type_ == 2:
        if any([validate_roman(x) for x in address.split()]):
            address = " ".join([ roman_to_int(x) if validate_roman(x) else x for x in address.split()])
        address__ = " ".join([x for x in clean_address(address).split() if x not in ["rt", "rw", "no", "nomor", "desa", "dusun"]])
        check_beginning = address__[0].isnumeric()
        while check_beginning:
            address__ = " ".join([x for x in address__.split()[1:]])
            check_beginning = address__[0].isnumeric()
        m = re.search(r"\d", address__)
        if m: 
            cluster = address__[:m.start()].strip()
            cluster = cluster.lower()
            return " ".join([x for x in standarized_address(cluster).split() if x not in ["rt", "rw", "no", "nomor", "desa"]]).title().strip()
        return " ".join(address__.split()[0:max(1, len(address.split())//5)]).title().strip()
    return address_.title()

def ospm(s1: str, s2: str) -> float:
    s1 = s1.replace(" ", "")
    s2 = s2.replace(" ", "")
    if len(s1) > len(s2):
        string_temp = s1
        s1 = s2
        s2 = string_temp
    length = len(s1) + len(s2)
    if not length:
        return 0
    if s1 == "" or s2 == "":
        return 0
    intersect = collections.Counter(s1) & collections.Counter(s2)
    matches = sum(intersect.values())
    GPM_difflib = 2.0 * matches / length
    if s1 in s2 and len(s1) > 3 and len(s1)/len(s2) > 0.4:
        return GPM_difflib + (1 - GPM_difflib) * (len(s1)/len(s2))
    return GPM_difflib

def ospm_sub(s1: str, s2: str) -> float:
    s1 = s1.replace(" ", "")
    s2 = s2.replace(" ", "")
    if len(s1) > len(s2):
        string_temp = s1
        s1 = s2
        s2 = string_temp
    length = len(s1) + len(s2)
    if not length:
        return 0
    if s1 == "" or s2 == "":
        return 0
    intersect = collections.Counter(s1) & collections.Counter(s2)
    matches = sum(intersect.values())
    GPM_difflib = 2.0 * matches / length
    if s1 in s2 and len(s1) > 3 and len(s1)/len(s2) > 0.4:
        return GPM_difflib + (1 - GPM_difflib) * (len(s1)/len(s2))
    return GPM_difflib

def contain_(s1: str, s2: str) -> float:
    s1 = s1.replace(" ", "")
    s2 = s2.replace(" ", "")
    if len(s1) > len(s2):
        string_temp = s1
        s1 = s2
        s2 = string_temp
    if s1 in s2 and len(s1) > 3:
        return 0.95
    return 0

def ospm_adj(s1: str, s2: str) -> float:
    s1 = s1.replace(" ", "")
    s2 = s2.replace(" ", "")
    if len(s1) > len(s2):
        string_temp = s1
        s1 = s2
        s2 = string_temp
    if len(s1)/len(s2) < 0.25 and len(s1) <= 3:
        return 0
    else:
        return ospm(s1, s2)

def sort_waybill(nodes):
    F = nx.Graph()
    nodes.reset_index(inplace=True)
    dict_waybill = nodes.name.to_dict()
    coordinates_ = [(x,y) for x,y in zip(nodes.loc[:, "latitude"], nodes.loc[:, "longitude"])]
    clusters = DBSCAN(eps=0.2/6371., min_samples=1, algorithm='ball_tree', metric='haversine').fit_predict(np.radians(nodes.loc[1:,["latitude", "longitude"]].values))
    permu = [x for x in itertools.combinations(nodes.name, 2)]
    dfs = pd.DataFrame(distance.cdist(coordinates_, coordinates_, haversine))
    dfs.rename(index=dict_waybill, inplace=True)
    dfs.rename(columns=dict_waybill, inplace=True)
    nodes_data = [(x["name"], x[1:].to_dict()) for index, x in nodes.iterrows()]
    F.add_nodes_from(nodes_data)

    # data = [[x,nodes.index[index+1]] for index, x in enumerate(nodes.index) if index < len(nodes.index)-1] + [[len(nodes)-1, 0]]
    # edges_data = pd.DataFrame(data=data, columns=["from", "to"])
    # edges = [(x["from"], x["to"], x[2:].to_dict()) for index, x in edges_data.iterrows()]
    # G.add_edges_from(edges)
    possible_edges = [x+tuple([{"weight": dfs.loc[x]}]) for x in permu]
    F.add_edges_from(possible_edges)
    pos = {data["name"]: (data["longitude"], data["latitude"]) for index, data in nodes.iterrows()} 
    cycle = nx_app.christofides(F, weight="weight")
    nodes["cluster"] = [0] + list(clusters + 1)
    nodes["rank"] = nodes.name.replace({x:i for i, x in enumerate(cycle[:-1])})
    nodes.sort_values(by="rank", ascending=True, inplace=True)
    edge_list = list(nx.utils.pairwise(cycle))
    total_distance_opt = sum([dfs.loc[x] for x in edge_list])
    

    # Draw route
    fig, ax = plt.subplots(1, 1)
    # fig.suptitle(f'Route comparison', fontweight="bold", fontsize=20)


    # # ax1
    # ax1.set_title('Current', size=16)
    # nx.draw_networkx(G, pos=pos, ax=ax1, edge_color="red")
    # ax1.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    # ax1.text(106.725, -6.132, f"total distance: {total_distance_current:.2f}", fontdict={"fontsize": 17})

    # ax2
    ax.set_title('Ranking', size=16)
    labels = {}
    for index, label in enumerate(list([x[0] for x in edge_list])):
        labels[label] = f"{index}"
        
    nx.draw_networkx(
        F,
        pos,
        with_labels=False,
        edgelist=edge_list,
        edge_color="green",
        ax=ax
    )
    nx.draw_networkx_labels(F,pos,labels, ax=ax)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    # ax3.set_title('Optimize', size=16)
    # nx.draw_networkx(
    #     F,
    #     pos,
    #     with_labels=False,
    #     edgelist=edge_list,
    #     edge_color="green",
    #     ax=ax3
    # )
    # nx.draw_networkx_labels(F,pos,labels, ax=ax3)
    # ax3.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    # ax3.text(106.725, -6.132, f"total distance: {total_distance_opt:.2f}", fontdict={"fontsize": 17})

    # ax4.set_title('Nextmv', size=16)
    # nx.draw_networkx(
    #     F,
    #     pos,
    #     with_labels=False,
    #     edgelist=edge_list_nextmv,
    #     edge_color="orange",
    #     ax=ax4
    # )
    # nx.draw_networkx_labels(F,pos,labels_nextmv, ax=ax4)
    # ax4.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    # ax4.text(106.725, -6.132, f"total distance: {total_distance_nextmv:.2f}", fontdict={"fontsize": 17})

    # # ax3
    # current = get_estimation_and_route(list(nodes.coordinates))
    # sections = current["routes"][0]["sections"]
    # durations = sum([x["summary"]["duration"] for x in sections])
    # lengths = sum([x["summary"]["length"] for x in sections])
    # ax1.text(106.725, -6.13, f"Durations: {durations/60:.2f} minute, \nLength: {lengths/1000:.2f} km", fontdict={"fontsize": 17})
    # coord = [fp.decode(x["polyline"]) for x in sections]
    # coord = [(y[1], y[0]) for x in coord for y in x]
    # xs, ys = zip(*coord) #create lists of x and y values
    # ax1.plot(xs,ys)
    # # nx.draw_networkx_nodes(G, pos=pos, ax=ax3)

    # # ax4
    # opt = get_estimation_and_route(list(nodes.coordinates[[x[0] for x in edge_list]]))
    # sections = opt["routes"][0]["sections"]
    # durations = sum([x["summary"]["duration"] for x in sections])
    # lengths = sum([x["summary"]["length"] for x in sections])
    # ax2.text(106.725, -6.13, f"Durations: {durations/60:.2f} minute, \nLength: {lengths/1000:.2f} km", fontdict={"fontsize": 17})
    # coord = [fp.decode(x["polyline"]) for x in sections]
    # coord = [(y[1], y[0]) for x in coord for y in x]
    # xs, ys = zip(*coord) #create lists of x and y values
    # ax2.plot(xs,ys)

    # ax3.text(106.725, -6.13, f"Durations: {durations/60:.2f} minute, \nLength: {lengths/1000:.2f} km", fontdict={"fontsize": 17})
    # ax3.plot(xs,ys)
    # # nx.draw_networkx_nodes(G, pos=pos, ax=ax4)

    # nextmv = get_estimation_and_route(list(nodes.coordinates[[x[0] for x in edge_list_nextmv]]))
    # sections = nextmv["routes"][0]["sections"]
    # durations = sum([x["summary"]["duration"] for x in sections])
    # lengths = sum([x["summary"]["length"] for x in sections])
    # ax4.text(106.725, -6.13, f"Durations: {durations/60:.2f} minute, \nLength: {lengths/1000:.2f} km", fontdict={"fontsize": 17})
    # coord = [fp.decode(x["polyline"]) for x in sections]
    # coord = [(y[1], y[0]) for x in coord for y in x]
    # xs, ys = zip(*coord) #create lists of x and y values
    # ax4.plot(xs,ys)

    # plt.show()
    return nodes.loc[:,["name", "rank", "cluster"]], total_distance_opt, fig


def sort_waybill_addrress(nodes, url):
    nodes.columns = ["waybill_no", "recipient_address"]
    # nodes["extracted_address"] = nodes.recipient_address.apply(lambda x: extract_address(x, url))
    map_extracted = extract_address_bulk(nodes.recipient_address.unique(), url)
    nodes["extracted_address"] = nodes.recipient_address.apply(lambda x: map_extracted[x])
    # data = {x:{**y, **{"full_address":w}} for w,x,y in zip(nodes.recipient_address, nodes.waybill_no, nodes.extracted_address)}
    # list_permutations = [x for x in itertools.combinations(data.keys(), 2)]
    df_group = nodes.groupby(["recipient_address"]).agg({"waybill_no": [list], "extracted_address": [pd.Series.mode]})
    df_group.reset_index(inplace=True)
    df_group.columns = ["recipient_address", "waybills", "extracted_address"]
    data = {x:{**y, **{"full_address":x, "waybills":z}} for x,y,z in zip(df_group.recipient_address, df_group.extracted_address, df_group.waybills)}
    list_permutations = [x for x in itertools.combinations(data.keys(), 2)]
    dict_similarity = {x: max(ospm(standarized_address(data[x[0]]['road_address']), 
                               standarized_address(data[x[1]]['road_address'])),
                          ospm(standarized_address(data[x[0]]['building_name_address']), 
                               standarized_address(data[x[1]]['building_name_address']))) for x in list_permutations}
    G = nx.Graph()
    G.add_nodes_from([(x, {**{"label": x}, **y}) for x,y in zip(df_group.recipient_address, df_group.extracted_address)])
    G.add_edges_from([x for x in list(dict_similarity.keys()) if dict_similarity[x]>0.75])
    list_group = list(nx.algorithms.components.connected_components(G))
    list_group = [list(x) for x in list_group]

    group_name = [[((standarized_address(data[y]['building_name_address']), 0), (standarized_address(data[y]['road_address']), 1), (data[y]['full_address'], 2)) for y in x] for x in list_group]
    mapping_group_name_type = [k for w in group_name for i in w for k in i]
    mapping_group_name_type = dict(sorted(mapping_group_name_type, key=lambda pair: (pair[0], -pair[1])))
    group_name = [sorted(collections.Counter([i[0] for y in x for i in y if i[0] != ""]).most_common(), 
                        key=lambda pair: (pair[1], -len(pair[0])), reverse=True) for x in group_name]
    group_name = [standarized_cluster_name(x[0][0], mapping_group_name_type[x[0][0]]) for x in group_name]
    mapping_cluster = {x:{} for x in group_name}
    list_group = [[w for y in x for w in data[y]["waybills"]] for x in list_group]
    
    for x,y in zip(group_name, list_group):
        mapping_cluster[x]["waybills"] = []
    for x,y in zip(group_name, list_group):
        mapping_cluster[x]["waybills"] = mapping_cluster[x]["waybills"] + y
    for x in mapping_cluster:
        mapping_cluster[x]["num_of_waybills"] = len(mapping_cluster[x]["waybills"])
    
    list_permutations_maps = [x for x in itertools.combinations(mapping_cluster.keys(), 2)]
    dict_similarity_maps = {x: ospm(standarized_address(x[0]), standarized_address(x[1])) for x in list_permutations_maps}
    G = nx.Graph()
    G.add_nodes_from([(x, {"label": x}) for x in mapping_cluster.keys()])
    G.add_edges_from([x for x in list(dict_similarity_maps.keys()) if dict_similarity_maps[x]>0.9])
    list_group_maps = list(nx.algorithms.components.connected_components(G))
    list_group_maps = [list(x) for x in list_group_maps]
    mapping_cluster_final = {max(x, key=lambda x: (-x.count(" "), -len(x))): {"waybills": [m for k in x for m in mapping_cluster[k]['waybills']]} for x in list_group_maps}
    for key, val in mapping_cluster_final.items():
        mapping_cluster_final[key]["num_of_waybills"] = len(val["waybills"])
    mapping_cluster_final = {x[2:].title() if "jl" in x[:3].lower() else x:y for x,y in mapping_cluster_final.items()}
    mapps = {i:x for x,y in mapping_cluster_final.items() for i in y["waybills"]}
    nodes["cluster"] = [mapps[x] for x in nodes.waybill_no]

    nodes["sub_cluster"] = nodes.waybill_no
    for cluster, data in mapping_cluster_final.items():
        df2 = nodes.loc[nodes.waybill_no.isin(data["waybills"])]
        df_group = df2.groupby(["recipient_address"]).agg({"waybill_no": [list], "extracted_address": [pd.Series.mode]})
        df_group.reset_index(inplace=True)
        df_group.columns = ["recipient_address", "waybills", "extracted_address"]
        data = {x:{**y, **{"full_address":x, "waybills":z}} for x,y,z in zip(df_group.recipient_address, df_group.extracted_address, df_group.waybills)}
        list_permutations = [x for x in itertools.combinations(data.keys(), 2)]
        if len(list_permutations) > 0:
            dict_similarity = {x: max(
                                        0,
                                        # mean([ospm(standarized_address(data[x[0]]['road_address']), 
                                        # standarized_address(data[x[1]]['road_address'])), 
                                        # ospm("".join((data[x[0]]['number_address']).lower()),
                                        # "".join((data[x[1]]['number_address']).lower()))]),
                                        ospm(standarized_address(clean_address(data[x[0]]['building_name_address'])), 
                                            standarized_address(clean_address(data[x[1]]['building_name_address'])))) for x in list_permutations}
            G = nx.Graph()
            G.add_nodes_from([(x, {**{"label": x}, **y}) for x,y in zip(df_group.recipient_address, df_group.extracted_address)])
            G.add_edges_from([x for x in list(dict_similarity.keys()) if dict_similarity[x]>0.8 or len(set(standarized_building(data[x[0]]['building_name_address']).split()) &
                                                                                                    set(standarized_building(data[x[1]]['building_name_address']).split())) > 1])
            list_group = list(nx.algorithms.components.connected_components(G))
            list_group = [list(x) for x in list_group]
            building_name = [(" ".join([standarized_building(data[y]['building_name_address']).upper() for y in x]), len(x)) for x in list_group]
            building_name = [{word: count for word, count in collections.Counter(x[0].split()).items() if count >= ((x[1]//2) + 1) and word.isalpha()} for x in building_name]
            group_name = [" ".join(x.keys()) for x in building_name]
            mapping_cluster_building = {x:{} for x in group_name}
            list_group = [[z for y in x for z in data[y]["waybills"]] for x in list_group]
            
            for x,y in zip(group_name, list_group):
                mapping_cluster_building[x]["waybills"] = []
            for x,y in zip(group_name, list_group):
                mapping_cluster_building[x]["waybills"] = mapping_cluster_building[x]["waybills"] + y
            for x in mapping_cluster_building:
                mapping_cluster_building[x]["num_of_waybills"] = len(mapping_cluster_building[x]["waybills"])
            
            list_permutations_maps = [x for x in itertools.combinations(mapping_cluster_building.keys(), 2)]
            dict_similarity_maps = {x: ospm(standarized_address(x[0]), standarized_address(x[1])) for x in list_permutations_maps}
            G = nx.Graph()
            G.add_nodes_from([(x, {"label": x}) for x in mapping_cluster_building.keys()])
            G.add_edges_from([x for x in list(dict_similarity_maps.keys()) if dict_similarity_maps[x]>0.9])
            list_group_maps = list(nx.algorithms.components.connected_components(G))
            list_group_maps = [list(x) for x in list_group_maps]
            mapping_cluster_final_building = {max(x, key=lambda x: (-x.count(" "), -len(x))): {"waybills": [m for k in x for m in mapping_cluster_building[k]['waybills']]} for x in list_group_maps}
            for key, val in mapping_cluster_final_building.items():
                mapping_cluster_final_building[key]["num_of_waybills"] = len(val["waybills"])
            mapping_cluster_final_building = {x[2:].title() if "jl" in x[:3].lower() else x:y for x,y in mapping_cluster_final_building.items()}
            mapps_building = {i:x.title() for x,y in mapping_cluster_final_building.items() for i in y["waybills"]}
            nodes["sub_cluster"] = nodes.sub_cluster.replace(mapps_building)
        else:
            mapps_building = {z:standarized_building(y["building_name_address"]).title() for x,y in data.items() for z in y["waybills"]}
            nodes["sub_cluster"] = nodes.sub_cluster.replace(mapps_building)
    nodes.sort_values(by=["cluster", "sub_cluster", "recipient_address"], inplace=True)

def cluster_waybill(nodes, url):
    nodes.columns = ["waybill_no", "recipient_address", "district", "city"]
    map_extracted = extract_address_bulk(nodes, url)
    nodes["clean_address"] = nodes["recipient_address"] + " " + nodes["district"] + " " + nodes["city"]
    nodes["extracted_address"] = nodes.clean_address.apply(lambda x: map_extracted[x])
    df_group = nodes.groupby(["recipient_address"]).agg({"waybill_no": [list], "extracted_address": [pd.Series.mode]})
    df_group.reset_index(inplace=True)
    df_group.columns = ["recipient_address", "waybills", "extracted_address"]
    data = {x:{**y, **{"full_address":x, "waybills":z}} for x,y,z in zip(df_group.recipient_address, df_group.extracted_address, df_group.waybills)}
    list_permutations = [x for x in itertools.combinations(data.keys(), 2)]
    # data = {x:{**y, **{"full_address":w}} for w,x,y in zip(nodes.recipient_address, nodes.waybill_no, nodes.extracted_address)}
    # list_permutations = [x for x in itertools.combinations(data.keys(), 2)]
    dict_similarity = {x: max(ospm(standarized_address(data[x[0]]['road_address']), 
                                standarized_address(data[x[1]]['road_address'])),
                            ospm(standarized_address(data[x[0]]['building_name_address']), 
                                standarized_address(data[x[1]]['building_name_address']))) for x in list_permutations}
    G = nx.Graph()
    G.add_nodes_from([(x, {**{"label": x}, **y}) for x,y in zip(df_group.recipient_address, df_group.extracted_address)])
    G.add_edges_from([x for x in list(dict_similarity.keys()) if dict_similarity[x] > 0.8])

    list_group = list(nx.algorithms.components.connected_components(G))
    list_group = [list(x) for x in list_group]

    group_name = [[((standarized_address(data[y]['road_address']), 0), (standarized_address(data[y]['building_name_address']), 1), (data[y]['full_address'], 2))*len(data[y]["waybills"]) for y in x] for x in list_group]
    mapping_group_name_type = [k for w in group_name for i in w for k in i]
    mapping_group_name_type = dict(sorted(mapping_group_name_type, key=lambda pair: (pair[0], -pair[1])))

    group_name_test = group_name

    group_name = [collections.Counter([i[0] for y in x for i in y if i[0] != ""]).most_common() for x in group_name_test]
    group_name = [sorted([(*y, mapping_group_name_type[y[0]]) for y in x],
                        key=lambda pair: (-pair[2], pair[1], -len(pair[0])), reverse=True) for x in group_name]
    group_name = [standarized_cluster_name(x[0][0], mapping_group_name_type[x[0][0]]) for x in group_name]

    mapping_cluster = {x:{} for x in group_name}
    list_group = [[w for y in x for w in data[y]["waybills"]] for x in list_group]

    for x,y in zip(group_name, list_group):
        mapping_cluster[x]["waybills"] = []
    for x,y in zip(group_name, list_group):
        mapping_cluster[x]["waybills"] = mapping_cluster[x]["waybills"] + y
    for x in mapping_cluster:
        mapping_cluster[x]["num_of_waybills"] = len(mapping_cluster[x]["waybills"])
        

    list_permutations_maps = [x for x in itertools.combinations(mapping_cluster.keys(), 2)]
    dict_similarity_maps = {x: max(ospm_adj(standarized_address(x[0]), standarized_address(x[1])),
                                contain_(standarized_address(x[0]), standarized_address(x[1]))) for x in list_permutations_maps}
    G = nx.Graph()
    G.add_nodes_from([(x, {"label": x}) for x in mapping_cluster.keys()])
    G.add_edges_from([x for x in list(dict_similarity_maps.keys()) if dict_similarity_maps[x]>0.85])
    list_group_maps = list(nx.algorithms.components.connected_components(G))
    list_group_maps = [list(x) for x in list_group_maps]
    mapping_cluster_final = {max(x, key=lambda x: (-x.count(" "), -len(x))): {"waybills": [m for k in x for m in mapping_cluster[k]['waybills']]} for x in list_group_maps}
    for key, val in mapping_cluster_final.items():
        mapping_cluster_final[key]["num_of_waybills"] = len(val["waybills"])
    mapping_cluster_final = {x[2:].title() if "jl" in x[:3].lower() else x:y for x,y in mapping_cluster_final.items()}
    mapps = {i:x for x,y in mapping_cluster_final.items() for i in y["waybills"]}
    nodes["cluster"] = [mapps[x] for x in nodes.waybill_no]
    nodes["sub_cluster"] = nodes.waybill_no
    for cluster, data in mapping_cluster_final.items():
        df2 = nodes.loc[nodes.waybill_no.isin(data["waybills"])]
        df_group = df2.groupby(["recipient_address"]).agg({"waybill_no": [list], "extracted_address": [pd.Series.mode]})
        df_group.reset_index(inplace=True)
        df_group.columns = ["recipient_address", "waybills", "extracted_address"]
        data = {x:{**y, **{"full_address":x, "waybills":z}} for x,y,z in zip(df_group.recipient_address, df_group.extracted_address, df_group.waybills)}
        list_permutations = [x for x in itertools.combinations(data.keys(), 2)]
        if len(list_permutations) > 0:
            dict_similarity = {x: min(
                ospm(standarized_address(clean_address(data[x[0]]['building_name_address'])), 
                                standarized_address(clean_address(data[x[1]]['building_name_address']))),
    #             mean([ospm(standarized_address(data[x[0]]['road_address']), 
    #                                        standarized_address(data[x[1]]['road_address'])), 
    #                                       ospm("".join((data[x[0]]['number_address']).lower()),
    #                                        "".join((data[x[1]]['number_address']).lower()))]),
                            jaro.jaro_winkler_metric(standarized_address(clean_address(data[x[0]]['building_name_address'])), 
                                standarized_address(clean_address(data[x[1]]['building_name_address'])))) for x in list_permutations}
            G = nx.Graph()
            G.add_nodes_from([(x, {**{"label": x}, **y}) for x,y in zip(df_group.recipient_address, df_group.extracted_address)])
            G.add_edges_from([x for x in list(dict_similarity.keys()) if dict_similarity[x]>=0.8 or len(set(standarized_building(data[x[0]]['building_name_address']).split()) &
                                                                                                    set(standarized_building(data[x[1]]['building_name_address']).split())) > 1])
            list_group_ = list(nx.algorithms.components.connected_components(G))
            list_group_ = [list(x) for x in list_group_]
            
            building_name = [(" ".join(sorted([standarized_building(data[y]['building_name_address']).upper() for y in x], key = lambda x: -len(x)))
                            , len(x)) for x in list_group_]
            build_names = [x[0] for x in building_name]
            building_name = [{word: count for word, count in collections.Counter(x[0].split()).items() if count >= ((x[1]//2) + 1) and word.isalpha()} for x in building_name]
            
            group_name = [" ".join(list(dict.fromkeys([l for l in y.split() if l in x.keys()]))) for x,y in zip(building_name, build_names)]
            mapping_cluster_building = {x:{} for x in group_name}
            
            list_group_ = [[z for y in x for z in data[y]["waybills"]] for x in list_group_]
            
            for x,y in zip(group_name, list_group_):
                mapping_cluster_building[x]["waybills"] = []
            for x,y in zip(group_name, list_group_):
                mapping_cluster_building[x]["waybills"] = mapping_cluster_building[x]["waybills"] + y
            for x in mapping_cluster_building:
                mapping_cluster_building[x]["num_of_waybills"] = len(mapping_cluster_building[x]["waybills"])
                
            if cluster.lower() == "jendral sudirman":
                print(build_names)
            
            list_permutations_maps = [x for x in itertools.combinations(mapping_cluster_building.keys(), 2)]
            dict_similarity_maps = {x: jaro.jaro_winkler_metric(standarized_building(x[0]), standarized_building(x[1]))
                                    if standarized_building(x[0]) != "" or standarized_building(x[1]) != "" else 0 for x in list_permutations_maps}
            G = nx.Graph()
            G.add_nodes_from([(x, {"label": x}) for x in mapping_cluster_building.keys()])
            G.add_edges_from([x for x in list(dict_similarity_maps.keys()) if dict_similarity_maps[x]>0.9])
            list_group_maps = list(nx.algorithms.components.connected_components(G))
            
            list_group_maps = [sorted(list(x), key=lambda x: -len(x)) for x in list_group_maps]
            mapping_cluster_final_building = {max(x, key=lambda x: (-x.count(" "), -len(x))): {"waybills": [m for k in x for m in mapping_cluster_building[k]['waybills']]} for x in list_group_maps}
            
            for key, val in mapping_cluster_final_building.items():
                mapping_cluster_final_building[key]["num_of_waybills"] = len(val["waybills"])
            mapping_cluster_final_building = {x[2:].title() if "jl" in x[:3].lower() else x:y for x,y in mapping_cluster_final_building.items()}
            mapps_building = {i:x.title() for x,y in mapping_cluster_final_building.items() for i in y["waybills"]}
            nodes["sub_cluster"] = nodes.sub_cluster.replace(mapps_building)
        else:
            mapps_building = {z:standarized_building(y["building_name_address"]).title() for x,y in data.items() for z in y["waybills"]}
            nodes["sub_cluster"] = nodes.sub_cluster.replace(mapps_building)
    return nodes.loc[:, [x for x in nodes.columns if x != "extracted_address"]]