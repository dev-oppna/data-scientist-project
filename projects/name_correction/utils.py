import re
import itertools
import difflib
import networkx as nx

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
        if any(toko in name for toko in LIST_NOT_NAME):
            continue
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
    dict_similarity = { x: difflib.SequenceMatcher(None, *x).ratio() for x in itertools.permutations(names_clean, 2)}
    G.add_nodes_from([(x, {"label": y}) for x,y in zip(names_clean, names)])
    G.add_edges_from([x for x in list(dict_similarity.keys()) if dict_similarity[x]>0.6])
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
    return dict_name[name], G