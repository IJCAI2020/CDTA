import load_data
import numpy as np
import networkx as nx
import scipy
import matplotlib as plt
import pickle
import load_data
from utils import *
# 1,5 college   seed = 1
# 0,9 email     seed = 3
# 0,2 wiki      seed=1
# 0,19 email_20 seed = 3
dataset = "email"
adjs = load_data.load_tn(dataset)
adj_1 = adjs[0]
adj_2 = adjs[9]
adj_all = np.zeros(adj_1.shape)
for adj in adjs:
    adj_all = adj_all + adj


# adjs = adjs[0:10]

# create_dataset_kg(adj_1, adj_2, seed=3, dataset=dataset)
# create_data_na(adj_1, adj_2, seed=3, dataset=dataset)
create_data_na_all(adj_1, adj_2, adj_all, seed=3, dataset=dataset)
# load_data.view_network(adjs)
# for i in range(len(adjs)):
#     write_edgelist_with_layer(adjs[i], dataset=dataset+str(i+1), layer_id=i+1)
