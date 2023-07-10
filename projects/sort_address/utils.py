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
    "apartment": "apartement"
}

def clean_address(address):
    if isinstance(address, str):
        address = address.lower()
        address = address.replace("alamat", "")
        address = address.replace(".", " ")
        address = address.replace(",", " ")
        address = address.replace("-", " ")
        address = address.replace("/", " ")
        address = re.sub('\(\s*(.*?)\s*\)', r'(\1)', address)
        address = re.sub(r"(\S)\(", r'\1 (', address)
        address = re.sub(r"\)(\S)", r') \1', address)
        address = re.sub('[^a-zA-Z0-9 \n\.]', '', address)
        address = re.sub(' +', ' ', address)
        address = address.strip()
        address = address.split(" ")
        address = [DICT_FIX.get(x,x) for x in address]
        return " ".join(address)
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
               "gang", "gedung"]
    address = address.lower()
    address = " ".join([x for x in address.split() if x not in mapping])
#     for key in mapping:
#         address = address.replace(key, ' ')
    if any([validate_roman(x) for x in address.split()]):
        road_address = " ".join([ roman_to_int(x) if validate_roman(x) else x for x in address.split()])
        address = road_address
    address = "".join([x for x in address if not x.isnumeric()])
    address = " ".join([x for x in address.split()])
    address = address.translate(str.maketrans('', '', string.punctuation))
    return address.lower().strip()

def standarized_cluster_name(address, type_):
    address_ = standarized_address(address)
    if len(address) > 15 and type_ == 2:
        m = re.search(r"\d", address)
        if m: 
            return address.title()[:m.start()].strip()
        return address.split()[0].title().strip()
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
    if s1 in s2:
        return max(GPM_difflib + (1 - GPM_difflib) * (len(s1)/len(s2)), 0.8)
    return GPM_difflib

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
    nodes.reset_index(inplace=True)
    nodes.columns = ["waybill_no", "recipient_address"]
    nodes["extracted_address"] = nodes.recipient_address.apply(extract_address, url)
    data = {x:{**y, **{"full_address":w}} for w,x,y in zip(nodes.recipient_address, nodes.waybill_no, nodes.extracted_address)}
    list_permutations = [x for x in itertools.combinations(data.keys(), 2)]
    dict_similarity = {x: max(ospm(standarized_address(data[x[0]]['road_address']), 
                               standarized_address(data[x[1]]['road_address'])),
                          ospm(standarized_address(data[x[0]]['building_name_address']), 
                               standarized_address(data[x[1]]['building_name_address']))) for x in list_permutations}
    G = nx.Graph()
    G.add_nodes_from([(x, {**{"label": x}, **y}) for x,y in zip(nodes.waybill_no, nodes.extracted_address)])
    G.add_edges_from([x for x in list(dict_similarity.keys()) if dict_similarity[x]>0.75])
    list_group = list(nx.algorithms.components.connected_components(G))
    list_group = [list(x) for x in list_group]
    group_name = [[((standarized_address(data[y]['building_name_address']), 0), (standarized_address(data[y]['road_address']), 1), (data[y]['full_address'], 2)) for y in x] for x in list_group]
    mapping_group_name_type = [k for w in group_name for i in w for k in i]
    mapping_group_name_type = dict(sorted(mapping_group_name_type, key=lambda pair: (pair[0], -pair[1])))
    group_name = [sorted(collections.Counter([i[0] for y in x for i in y if i[0] != ""]).most_common(), 
                        key=lambda pair: (pair[1], -len(pair[0])), reverse=True) for x in group_name]
    group_name = [[((standarized_address(data[y]['building_name_address']), 0), (standarized_address(data[y]['road_address']), 1), (data[y]['full_address'], 2)) for y in x] for x in list_group]
    mapping_group_name_type = [k for w in group_name for i in w for k in i]
    mapping_group_name_type = dict(sorted(mapping_group_name_type, key=lambda pair: (pair[0], -pair[1])))
    group_name = [sorted(collections.Counter([i[0] for y in x for i in y if i[0] != ""]).most_common(), 
                        key=lambda pair: (pair[1], -len(pair[0])), reverse=True) for x in group_name]
    group_name = [standarized_cluster_name(x[0][0], mapping_group_name_type[x[0][0]]) for x in group_name]
    mapping_cluster = {x:{} for x in group_name}
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
    nodes.sort_values(by=["cluster"], inplace=True)
    return nodes