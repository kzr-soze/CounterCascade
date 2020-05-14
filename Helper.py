# %matplotlib inline
import matplotlib.pyplot as plt
from random import uniform, seed
import numpy as np
import numpy.random as rand
import pandas as pd
import time
# from igraph import *
import random
from collections import Counter
import operator

from heapq import heappush, heappop
from itertools import count

import networkx as nx

def network_transform(G,S_r,seed_name,p=0.5):
    G_original = G.copy()
    for e in G_original.edges:
        G.edges[e]['weight'] = -np.log(p)
    new_edges = {}
    previous = len(S_r)
    G.add_node(seed_name)
    for r in S_r:
        for n in G_original[r].keys():
            e = (seed_name,n)
            if e in new_edges.keys():
                new_edges[e] = 1 - (1-p) * (1-new_edges[e])
            elif n not in S_r:
                new_edges[e] = p
        G.remove_node(r)
    for e in new_edges.keys():

        G.add_edge(e[0],e[1],weight=-np.log(new_edges[e]))
        G.add_edge(e[1],e[0],weight=-np.log(new_edges[e]))
    return G,G_original

def get_l_path_set(G,n,l):
    path_set = []
    path_weights = {}
    for i in G.nodes:
        if not i == n:
            lengths,paths = k_shortest_paths(G,n,i,k=l,weight='weight')
            path_set+=paths
            for j in range(len(paths)):
                path_weights[repr(paths[j])] = np.exp(-lengths[j])
    print("Path weights computed")
    return path_set,path_weights

def get_rand_inst(G):
    G_new = G.copy()
    for e in G.edges:
        if e[0] < e[1] and np.random.uniform(0,1) > np.exp(-G_new.edges[e]['weight']):
            # print(e)
            G_new.remove_edge(e[0],e[1])
            G_new.remove_edge(e[1],e[0])
    return(G_new)

def get_blocked_paths(path_set,R_r,R_b,seed_r,seed_b,blocked_instances,trials):
    red_set,blue_set = get_blue_red_sets(R_r,R_b,seed_r,seed_b)
    bs = set(blue_set)
    for p in path_set:
        if bs.intersection(p):
            blocked_instances[seed_b][repr(p)] += 1.0/trials

def get_best_blocker(G,path_weights,block_set):
    best_score = 0
    best_node = -1
    for i in G.nodes:
        block_likelihood = block_set[i]
        val = 0
        for j in path_weights.keys():
            val += path_weights[j]*block_likelihood[j]
        if val >= best_score:
            best_score = val
            best_node = i
    return best_node,best_score

def greedy_k_best_blockers(G,path_weights,block_set,k):
    blockers = []
    score = 0
    for i in range(k):
        n,marginal_val = get_best_blocker(G,path_weights,block_set)
        if n >-1:
            blockers.append(n)
            block_likelihood = block_set[n]
            for j in path_weights.keys():
                block_likelihood[j] = 0
            score+= marginal_val
            for j in path_weights.keys():
                path_weights[j] = path_weights[j] * (1-block_set[n][j])

    return blockers,score

def k_shortest_paths(G, source, target, k=1, weight='weight'):
    """Returns the k-shortest paths from source to target in a weighted graph G.
    Parameters
    ----------
    G : NetworkX graph
    source : node
       Starting node
    target : node
       Ending node

    k : integer, optional (default=1)
        The number of shortest paths to find
    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight
    Returns
    -------
    lengths, paths : lists
       Returns a tuple with two lists.
       The first list stores the length of each k-shortest path.
       The second list stores each k-shortest path.
    Raises
    ------
    NetworkXNoPath
       If no path exists between source and target.
    Examples
    --------
    >>> G=nx.complete_graph(5)
    >>> print(k_shortest_paths(G, 0, 4, 4))
    ([1, 2, 2, 2], [[0, 4], [0, 1, 4], [0, 2, 4], [0, 3, 4]])
    Notes
    ------
    Edge weight attributes must be numerical and non-negative.
    Distances are calculated as sums of weighted edges traversed.
    """
    if source == target:
        return ([0], [[source]])

    length, path = nx.single_source_dijkstra(G, source, weight=weight)
#     print(length,path)
    if target not in length:
        print("node %s not reachable from %s" % (target, source))
        return [],[]
        # raise nx.NetworkXNoPath("node %s not reachable from %s" % (source, target))

    lengths = [length[target]]
    paths = [path[target]]
    c = count()
    B = []
    G_original = G.copy()

    for i in range(1, k):
        for j in range(len(paths[-1]) - 1):
            spur_node = paths[-1][j]
            root_path = paths[-1][:j + 1]

            edges_removed = []
            for c_path in paths:
                if len(c_path) > j and root_path == c_path[:j + 1]:
                    u = c_path[j]
                    v = c_path[j + 1]
                    if G.has_edge(u, v):
                        edge_attr = G.edges[u,v]
                        G.remove_edge(u, v)
                        edges_removed.append((u, v, edge_attr))

            for n in range(len(root_path) - 1):
                node = root_path[n]
                for u in G.nodes:
                    if [u,node] in G.edges:
                        edge_attr = G.edges[u,node]
                        G.remove_edge(u, node)
                        edges_removed.append((u, node, edge_attr))
                    if [node,u] in G.edges:
                        edge_attr = G.edges[node,u]
                        G.remove_edge(node,u)
                        edges_removed.append((node,u, edge_attr))
            spur_path_length, spur_path = nx.single_source_dijkstra(G, spur_node, weight=weight)
            if target in spur_path and spur_path[target]:
                total_path = root_path[:-1] + spur_path[target]
                total_path_length = get_path_length(G_original, root_path, weight) + spur_path_length[target]
                heappush(B, (total_path_length, next(c), total_path))

            for e in edges_removed:
                u, v, edge_attr = e
                if weight:
                    G.add_edge(u, v, weight = edge_attr[weight])

        if B:
            (l, _, p) = heappop(B)
            lengths.append(l)
            paths.append(p)
        else:
            break
    G = G_original

    return (lengths, paths)

def get_path_length(G, path, weight='weight'):
    length = 0
    if len(path) > 1:
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]

            length += G.edges[u,v].get(weight, 1)

    return length

def get_blue_red_sets(R_r,R_b,seed_r,seed_b):
    lengths_b,paths_b = nx.single_source_dijkstra(R_b,seed_b,weight=None)
    lengths_r,paths_r = nx.single_source_dijkstra(R_r,seed_r,weight=None)
    blue_set = []
    red_set = []
    index_blue = 0
    index_red = 0
    nodes_b = list(lengths_b.keys())
    nodes_r = list(lengths_r.keys())
    ib = 0
    nb = nodes_b[ib]
    ir = 0
    nr = nodes_r[ir]
    go = True
    prev_nb = -1
    prev_nr = -1
    while go:
        if nr == prev_nr and nb == prev_nb:
            raise nx.NetworkXNoPath("problem")
        prev_nr = nr
        prev_nb = nb
        flag_b = False #True if nb is processed
        flag_r = False #True if nr is processed

        # nb not in nodes_r and closer to R_b than nr is to R_r, implying nb is blue
        if nb not in nodes_r and lengths_b[nb] <= lengths_r[nr]:
            # print(1)
            blue_set.append(nb)
            flag_b = True
        # nr not in nodes_b and closer to R_r than nb is to R_b, implying nr is red
        elif nr not in nodes_b and lengths_r[nr] <= lengths_b[nb]:
            # print(2)
            red_set.append(nr)
            flag_r = True
        # nb in nodes_r but closer to R_b than nr is to R_r, implying nb is blue as existing path is uncut
        elif nb in nodes_r and lengths_b[nb] < lengths_r[nb]:
            # print(3)
            blue_set.append(nb)
            flag_b = True
            flag_r = True
            R_r.remove_node(nb)
            lengths_r,paths_r = nx.single_source_dijkstra(R_r,seed_r,weight=None)
            nodes_r = list(lengths_r.keys())
            nodes_r = [n for n in nodes_r if n not in red_set]
            ir = -1
        # nr in nodes_b but closer to R_r than nb is to R_b, implying nr is red as existing path is uncut
        elif nr in nodes_b and lengths_r[nr] < lengths_b[nr]:
            # print(4)
            red_set.append(nr)
            flag_r = True
            flag_b = True
            R_b.remove_node(nr)
            lengths_b,paths_b = nx.single_source_dijkstra(R_b,seed_b,weight=None)
            nodes_b = list(lengths_b.keys())
            nodes_b = [n for n in nodes_b if n not in blue_set]
            ib = -1

        # Both colors try to influence a node simultaneously
        elif (nb in nodes_r and lengths_r[nb] == lengths_b[nb]):
            # print(5)
            e_b=(paths_b[nb][-2],paths_b[nb][-1])
            e_r=(paths_r[nb][-2],paths_r[nb][-1])
            pb = np.exp(-R_b.edges[e_b]['weight'])
            pr = np.exp(-R_r.edges[e_r]['weight'])

            # Red successfully influences
            if rand.random() <= pr/(pr+pb):
                flag_b = True
                R_b.remove_node(nb)
                lengths_b,paths_b = nx.single_source_dijkstra(R_b,seed_b,weight=None)
                nodes_b = list(lengths_b.keys())
                nodes_b = [n for n in nodes_b if n not in blue_set]
                ib = -1
            # Blue successfully influences
            else:
                blue_set.append(nb)
                flag_b = True
                flag_r = True
                R_r.remove_node(nb)
                lengths_r,paths_r = nx.single_source_dijkstra(R_r,seed_r,weight=None)
                nodes_r = list(lengths_r.keys())
                nodes_r = [n for n in nodes_r if n not in red_set]
                ir = -1

        elif (nr in nodes_b and lengths_b[nb] == lengths_r[nb]):
            e_b=(paths_b[nb][-2],paths_b[nb][-1])
            e_r=(paths_r[nb][-2],paths_r[nb][-1])
            pb = np.exp(-R_b.edges[e_b]['weight'])
            pr = np.exp(-R_r.edges[e_r]['weight'])

            # Red successfully influences
            if rand.random() <= pr/(pr+pb):
                red_set.append(nr)
                flag_r = True
                flag_b = True
                R_b.remove_node(nr)
                lengths_b,paths_b = nx.single_source_dijkstra(R_b,seed_b,weight=None)
                list(nodes_b = lengths_b.keys())
                nodes_b = [n for n in nodes_b if n not in blue_set]
                ib = -1
            # Blue successfully influences
            else:
                flag_r = True
                R_r.remove_node(nb)
                lengths_r,paths_r = nx.single_source_dijkstra(R_r,seed_r,weight=None)
                nodes_r = list(lengths_r.keys())
                nodes_r = [n for n in nodes_r if n not in red_set]
                ir = -1

        if flag_r:
            ir += 1
            if ir >= len(nodes_r):
                go = False
                blue_set = list(set(blue_set+nodes_b))
            else:
                nr = nodes_r[ir]
        if flag_b:
            ib += 1
            if ib >= len(nodes_b):
                go = False
                red_set = list(set(red_set+nodes_r))
            else:
                nb = nodes_b[ib]
    return red_set,blue_set
