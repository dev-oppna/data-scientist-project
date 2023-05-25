import numpy as np
import networkx as nx
from haversine import haversine
import itertools
from scipy.spatial import distance
import networkx.algorithms.approximation as nx_app
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

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