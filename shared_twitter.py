#!/usr/bin/env python
# coding: utf-8

# In[39]:
import matplotlib
matplotlib.use('Agg')



import csv
import pandas as pd
import time
import sys
import itertools
from datetime import datetime
import sys
import networkx as nx
import plotly as py
import plotly.graph_objects as go
from plotly.graph_objs import *
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from scipy.stats.stats import pearsonr
import community
import matplotlib.cm as cm
import random

# In[40]:


from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans


# In[41]:


from functions import *


# In[52]:


dates = ['2020-03-28', '2020-03-29', '2020-03-30', '2020-03-31',
        '2020-04-01', '2020-04-02', '2020-04-03', '2020-04-04',
        '2020-04-05', '2020-04-06', '2020-04-07', '2020-04-08',
         '2020-04-09', '2020-04-10']
top_users_display = 10
filter_by_top_cluster = True
top_clusters_display=10
param_subset=False
param_n_subset=1000


# In[53]:


#Returns a list of shared users from dictionnary of dataframes
def get_common_users(df_dict):
    l_common_users = []
    set_common_users = set(df_dict[list(df_dict.keys())[0]].user_retweeted)
    for key in list(df_dict.keys()):
        set_df_i = set(df_dict[key].user_retweeted)
        set_common_users = set.intersection(set_common_users, set_df_i)
    return list(set_common_users)


# In[54]:


#Returns dictionnary of dataframe where retweeted users are shared to every dataframes
def filter_dataframe_by_common_users(df_dict, l_common):
    for key in list(df_dict.keys()):
        df = df_dict[key]
        df_filter = df[df.user_retweeted.isin(l_common)]
        df_dict[key] = df_filter
    return df_dict


# In[56]:


if __name__=='__main__':
    df_dict = load_folder_as_dict(dates, prefix='../data/', subset=param_subset, n_subset=param_n_subset)
    l_common = get_common_users(df_dict)
    df_dict = filter_dataframe_by_common_users(df_dict, l_common)
    graph_dict = define_graphs(df_dict)
    print('Range of dates:', dates)

    for key in list(df_dict.keys()):
        t1 = time.time()
        print('Date: ', str(key).split('_')[-1])

        df = df_dict[key]
        G = graph_dict[key]

        #Keep connected component
        G, df = get_df_G_connected_comp(G, df)
        G_und = G.to_undirected()

        #define df_nodes
        df_nodes = define_df_nodes(df, G)
        #Reassign top cluster for a better display
        if filter_by_top_cluster:
            df_nodes = reassign_top_clusters(df_nodes, top_clusters=top_clusters_display)
            df_nodes.loc[:, 'cluster_id'] = df_nodes.loc[:, 'new_cluster_id']
        savename_nodes  = '../shared_users/processed_data/nodes/' + str(key).split('_')[-1]+'_nodes.csv'
        df_nodes.to_csv(savename_nodes,  encoding='utf-8')
        
        #define df_edges
        df_edges = define_df_edges(df, G)
        savename_edges  = '../shared_users/processed_data/edges/' + str(key).split('_')[-1]+'_edges.csv'
        df_edges.to_csv(savename_edges,  encoding='utf-8')

        #compute and save figures according to clustering and (top) most important nodes
        savename_clusters  = '../shared_users/visualisation/clusters/' + str(key).split('_')[-1]+'_clusters'
        vizualize_from_df(df_nodes, G_und, savename=savename_clusters)
        savename_clusters  = '../shared_users/visualisation/betweenness/' + str(key).split('_')[-1]+'_betweenness'
        vizualize_from_df_betweenness(df_nodes, G_und, top=top_users_display ,savename=savename_clusters)
        savename_clusters  = '../shared_users/visualisation/closeness/' + str(key).split('_')[-1]+'_closeness'
        vizualize_from_df_closeness(df_nodes, G_und, top=top_users_display ,savename=savename_clusters)
        savename_clusters  = '../shared_users/visualisation/pagerank/' + str(key).split('_')[-1]+'_pagerank'
        vizualize_from_df_pagerank(df_nodes, G_und, top=top_users_display, savename=savename_clusters)
        
        t2 = time.time()
        delta_t = t2-t1
        print('Time taken (sec):', delta_t)


# In[ ]:




