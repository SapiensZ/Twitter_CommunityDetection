#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Network Science Analytics
Lab 2: Community Detection
February 28, 2020
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import networkx as nx
from scipy.cluster import hierarchy

def plot_dendrogram(G, partitions):

    num_of_nodes = G.number_of_nodes()
    dist = np.ones( shape=(num_of_nodes, num_of_nodes), dtype=np.float )*num_of_nodes
    d = num_of_nodes-1
    for partition in partitions:
        for subset in partition:
            for i in range(len(subset)):
                for j in range(i+1, len(subset)):
                    subsetl = list(subset)

                    dist[int(subsetl[i]), int(subsetl[j])] = d
                    dist[int(subsetl[j]), int(subsetl[i])] = d
        d -= 1



    dist_list = [dist[i,j] for i in range(num_of_nodes) for j in range(i+1, num_of_nodes)]


    Z = hierarchy.linkage(dist_list, 'complete')
    plt.figure()
    dn = hierarchy.dendrogram(Z)

def visualize(G, partition1, partition2):
    colors=['r', 'b', 'g', 'm', 'y', 'k']
    if len(partition1) > len(colors) or len(partition2) > len(colors):
        raise ValueError("There is no enough colors, you can add more colors to the list!")
    
    
    pos=nx.spring_layout(G) # positions for all nodes
    
    plt.subplot(1, 2, 1)
    for commId, nodeset in enumerate(partition1):
        nx.draw_networkx_nodes(G, pos, nodelist=nodeset, node_color=colors[commId], node_size=100, alpha=0.4)
    nx.draw_networkx_edges(G, pos, alpha=0.5)

    plt.subplot(1, 2, 2)
    for commId, nodeset in enumerate(partition2):
        nx.draw_networkx_nodes(G, pos, nodelist=nodeset, node_color=colors[commId], node_size=100, alpha=0.4)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    plt.show()