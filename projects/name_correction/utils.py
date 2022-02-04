import re
import itertools
import difflib
import networkx as nx
import plotly.graph_objects as go
from pyvis import network as net

LIST_PANGGILAN = ["bapak", "bapa", "bpk", "ibu", "bu", "mas", "mba", "shop", "sdr", "up ", "sdri", "sdra",
                 "mang", "kang", "teh"]

LIST_NOT_NAME = ["pt ", "ud ", "toko", "shop", "official", "oficial", "manajer", "manager", "kepala", 
                 "head", "lead", "cv ", "tukang", "poli", "klinik", "store"]


def len_group(set_of_group):
    return (len(set_of_group), sum([len(x) for x in set_of_group])/len(set_of_group))

def clear_name(word):
    if "(" in word:
        return False
    elif word in LIST_PANGGILAN:
        return False
    elif len(word) < 2:
        return False
    if any(toko in word for toko in LIST_NOT_NAME):
        return False
    elif word.isalpha():
        return True
    return True

def strip_name(name):
    name = name.lower()
    names = name.split("/")
    for name in names:
        name = name.split(" atau ")[0]
        name = name.replace(".", " ")
        name = ' '.join(name.split())
        name = re.sub('\(\s*(.*?)\s*\)', r'(\1)', name)
        name = re.sub(r"(\S)\(", r'\1 (', name)
        in_bracket = re.findall("\((.*?)\)", name)
        if len(in_bracket) > 0:
            for k in in_bracket:
                for j in strip_name(k):
                    yield j
        name = re.sub("[\(\[].*?[\)\]]", "", name)
        name = name.split(" ")
        name = [word for word in name if clear_name(word)]
        name = [''.join([i for i in word if not i.isdigit()]) for word in name]
        name = [word for word in name if set('aeiou').intersection(word.lower())]
        name = " ".join(name)
        name = re.sub('[^a-zA-Z0-9 \n\.]', '', name)
        name = name.strip()
        name = name.lower()
        if name != "":
            yield name.title()

def get_name(list_name):
    G = nx.Graph()
    names = [strip_name(x) for x in list_name]
    names = [x for m in names for x in m ]
    names_clean = list(map(lambda x: x[1] + str(names[:x[0]].count(x[1]) + 1) if names.count(x[1]) > 1 else x[1], enumerate(names)))
    names_clean = [x.lower().replace(" ", "") for x in names_clean]
    dict_name = {x: y for x,y in zip(names_clean, names)}
    list_permutations = [x for x in itertools.combinations(names_clean, 2)]
    list_permutations_clean = [(dict_name[x[0]].lower(), dict_name[x[1]].lower()) for x in list_permutations]
    dict_similarity = {x: difflib.SequenceMatcher(None, *y).ratio() for x,y in zip(list_permutations, list_permutations_clean)}
    G.add_nodes_from([(x, {"label": y}) for x,y in zip(names_clean, names)])
    G.add_edges_from([x for x in list(dict_similarity.keys()) if dict_similarity[x]>0.5])
    list_group = list(nx.algorithms.components.connected_components(G))
    bigest_group = max(list_group, key=len_group)
    bigest_group_with_deg = [x for x in dict(G.degree()).items() if x[0] in bigest_group]
    name = max(bigest_group, key=len)
    name_deg = max(bigest_group_with_deg, key = lambda x : x[1])[0]

    max_deg = max([x[1] for x in bigest_group_with_deg])
    names_to_pick = [x[0] for x in bigest_group_with_deg if x[1] == max_deg]

    if dict_name[name].count(" ") <= dict_name[name_deg].count(" "):
        if len(names_to_pick) > 1:
            name = max(names_to_pick, key=len)
        else:
            name = names_to_pick[0]
    conf_level = [y for x,y in dict_similarity.items() if name in x and y > 0.5]
    conf_level = int(sum(conf_level)/len(conf_level)*100)
    return dict_name[name], G, conf_level

def make_graph(nx_graph):
    g4 = net.Network(height='400px', width='700px', notebook=True)
    g4.from_nx(nx_graph)
    nodes, edges, heading, height, width, options = g4.get_network_data()
    html = g4.template.render(height=height,
                                width=width,
                                nodes=nodes,
                                edges=edges,
                                heading=heading,
                                options=options,
                                physics_enabled=g4.options.physics.enabled,
                                use_DOT=g4.use_DOT,
                                dot_lang=g4.dot_lang,
                                widget=g4.widget,
                                bgcolor=g4.bgcolor,
                                conf=g4.conf,
                                tooltip_link=True)
    return html

def make_gauge(value):
    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = value,
        mode = "gauge+number",
        title = {'text': "Confidence"},
        gauge = {'axis': {'range': [None, 100]},
                'steps' : [
                    {'range': [0, 50], 'color': "lightpink"},
                    {'range': [50, 75], 'color': "lightyellow"},
                    {'range': [75, 100], 'color': "lightgreen"}],
                'threshold' : {'line': {'color': "blue", 'width': 4}, 'thickness': 0.75, 'value': 95}}))
    return fig