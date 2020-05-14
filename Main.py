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
# import csv

from Helper import *

l= 3
trials = 5
k = 20
est_spread = 5
p_r = 0.5
p_b = 0.15
S_r = [236,186]
l = 25
w = 25

if p_b < 0.00001:
    trials = 1

graph = 2
if graph == 1:
    G = nx.karate_club_graph()
    G = nx.DiGraph(G)
    S_r = [1,5]
elif graph ==2:
    G = nx.read_edgelist("0.edges", nodetype=int)
    G = nx.DiGraph(G)
    S_r = [236,186]
    print(G.nodes)
    print(322 in G.nodes)
elif graph == 3:
    G = nx.grid_graph([l,w])
    count = 0
    mapping = {}
    for i in G.nodes.keys():
        mapping[i] = count
        count+=1
    S = []
    for i in S_r:
        S.append(l * (i[0]-1) + i[1]-1)
    S_r = S
    G = nx.relabel_nodes(G,mapping)
    print(G)
    print(G[0].keys())
    G = nx.DiGraph(G)
else:
    print('Invalid graph choice')

def run_program(G,S_r,k,l,p_r,p_b):
    n = len(G.nodes)
    temp = []
    for i in G.nodes:
        temp.append(i)
    temp.sort()
    print(temp)
    print(n)
    seed_r = 2*n
    seed_b = seed_r+1
    go = True
    while go:
        if seed_b in S_r:
            seed_b+=1
        else:
            go = False

    # Get transformed networks with conjoined nodes and appropriate edge weights
    G_red,G = network_transform(G,S_r,seed_r,p=p_r)
    G_blue = nx.DiGraph(G_red.copy())

    for e in G_blue.edges:
        G_blue.edges[e]['weight'] = -np.log(p_b)
        G_blue.edges[e[1],e[0]]['weight'] = -np.log(p_b)

    for i in S_r:
        if i in G_blue.nodes:
            G_blue.remove_node(i)
    G_blue.remove_node(seed_r)

    # Get path set for covering
    path_set,path_weights = get_l_path_set(G_red,seed_r,l)
    p2 = []
    for p in path_set:
        if p not in p2:
            p2.append(p)
    path_set=p2
    block_set = {}
    for node in G_blue.nodes:
        block_set[node] = {}
        for p in path_weights.keys():
            block_set[node][p] = 0

    count = 0
    for node in G_blue.nodes:
        count+=1
        print('\n\n')
        print('Node {}  of {}\n'.format(count,len(G_blue.nodes)))
        for t in range(trials):
            print("Estimating blockers, Node: {}  Trial: {} of {}".format(node,t+1,trials))
            R_r = get_rand_inst(G_red)
            R_b = get_rand_inst(G_blue)
            get_blocked_paths(path_set,R_r,R_b,seed_r,node,block_set,trials)

    # Estimate coverage without interdiction
    original_red = 0
    for i in range(est_spread):
        print("Estimating unimpeded spread, Trial {} of {}".format(i+1,est_spread))
        R_r = get_rand_inst(G_red)
        length,paths = nx.single_source_dijkstra(R_r,seed_r,weight=None)
        original_red += (1.0*len(length.keys())+len(S_r)-1)/est_spread

    # Estimate performance of greedy solution
    S_b,score = greedy_k_best_blockers(G_blue,path_weights,block_set,k)
    print("S_b: ",S_b)
    G_blue, G_red = network_transform(G_red,S_b,seed_b,p=p_b)
    for i in S_b:
        G_red.remove_node(i)
    G_blue.remove_node(seed_r)
    expected_red = 0
    expected_blue = 0
    for i in range(est_spread):
        print("Estimating impeded spread, Trial {} of {}".format(i+1,est_spread))
        R_r = get_rand_inst(G_red)
        R_b = get_rand_inst(G_blue)
        red_set,blue_set = get_blue_red_sets(R_r,R_b,seed_r,seed_b)
        expected_red += (1.0*len(red_set) + len(S_r)-1)/est_spread
        expected_blue += (1.0*len(blue_set) + k-1)/est_spread

    return S_b, original_red, expected_red, expected_blue

# S_b = [315,322,343,122,346,214,53,119,236,20,299,149,204,180,73,226,194,126,87,92]
def test_k_range(G,S_b,S_r,k,p_r,p_b):
    n = len(G.nodes)
    print(G.nodes)
    print(n)
    seed_r = n
    seed_b = 0
    go = True
    while go:
        if seed_b in S_r:
            seed_b+=1
        else:
            go = False

    # Get transformed networks with conjoined nodes and appropriate edge weights
    G_red,G = network_transform(G,S_r,seed_r,p=p_r)
    G_blue = nx.DiGraph(G_red.copy())

    for e in G_blue.edges:
        G_blue.edges[e]['weight'] = -np.log(p_b)
        G_blue.edges[e[1],e[0]]['weight'] = -np.log(p_b)

    for i in S_r:
        G_blue.remove_node(i)
    G_blue.remove_node(n)

    # Estimate coverage without interdiction
    original_red = 0
    for i in range(est_spread):
        print("Estimating unimpeded spread, Trial {} of {}".format(i+1,est_spread))
        R_r = get_rand_inst(G_red)
        length,paths = nx.single_source_dijkstra(R_r,seed_r,weight=None)
        original_red += (1.0*len(length.keys())+len(S_r)-1)/est_spread

    # Estimate performance of greedy solution
    seed_b = n+1
    er = np.zeros([k+1,2])
    er[0,0] = original_red
    G_red0 = G_red.copy()
    for t in range(k):
        S_b0 = S_b[0:t+1]
        print(S_b0)
        G_red = G_red0.copy()
        G_blue, G_red = network_transform(G_red,S_b0,seed_b,p=p_b)

        for i in S_b0:
            G_red.remove_node(i)
        G_blue.remove_node(seed_r)
        expected_red = 0
        expected_blue = 0
        for i in range(est_spread):
            print("Estimating {}-impeded spread, Trial {} of {}".format(t+1,i+1,est_spread))
            R_r = get_rand_inst(G_red)
            R_b = get_rand_inst(G_blue)
            red_set,blue_set = get_blue_red_sets(R_r,R_b,seed_r,seed_b)
            expected_red += (1.0*len(red_set) + len(S_r)-1)/est_spread
            expected_blue += (1.0*len(blue_set) + t-1)/est_spread
        er[t+1,0] = expected_red
        er[t+1,1] = expected_blue
    return er

S_b, original_red, expected_red, expected_blue = run_program(G,S_r,k,l,p_r,p_b)

print(S_b)
print(original_red)
print(expected_red)
print(expected_blue)

# S_b = [315,322,343,122,346,214,53,119,236,20,299,149,204,180,73,226,194,126,87,92]
# S_b = [167,204,379,165,377,253,403,353,191,141]
# S_b = [33,1,32,6,3,31,10,27,8,7,13,5,11,12,21,17,9,19,4,28]
# er = test_k_range(G,S_b,S_r,k,p_r,p_b)
# print(er)
# np.savetxt("foo.csv",er,delimiter=",")
