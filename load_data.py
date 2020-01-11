import scipy.sparse as sp
import torch
import numpy as np
import pickle


def view_network(adjs):
    for index, adj in enumerate(adjs):
        print(index,np.array(np.array(adj).nonzero()).shape)

    temp = np.full(adjs[0].shape, 0)
    out = []
    for i in range(len(adjs)):
        temp += adjs[i].astype(int)
    for i in range(len(adjs)):
        out.append((np.array(np.where(temp==i)).shape))
    print(out)


def load_tn_wt(dataset):

    edge_file = open("data/{}.edge".format(dataset), 'r')
    edges = edge_file.readlines()
    node_num = int(edges[0].split('\t')[1].strip())
    edge_num = int(edges[1].split('\t')[1].strip())
    print("node number:{} , edge_number:{}".format(node_num, edge_num))
    edges.pop(0)
    edges.pop(0)

    max = 0
    max_node = 0
    min =1082439619
    for line in edges:
        timestamp = int(line.split(' ')[2].strip())
        node = int(line.split(' ')[1].strip())
        if timestamp > max:
            max = timestamp
        if timestamp < min:
            min = timestamp
        if node > max_node:
            max_node = node

    period = 20
    print("max_node:"+str(max_node))
    if dataset == "email1" or dataset == "email" or dataset == "email_all":
        period = 40

    gap = float((max+1-min)/period)

    print("min_time:{},max_time{},period:{},gap:{}".format(str(min),
                                                           str(max),
                                                           str(period),
                                                           str(gap)))

    shape = [period, node_num, node_num]
    adjs = np.zeros(shape)
    index = 0
    for line in edges:
        node1 = int(line.split(' ')[0].strip())-1
        node2 = int(line.split(' ')[1].strip())-1
        timestamp = int(line.split(' ')[2].strip())
        time = int((timestamp-min)/gap)
        adjs[time][node1][node2] = 1.0
        print(index, time)
        index = index+1

    save_file_name = dataset+'.edge'
    # with open(save_file_name, 'wb') as fp:
    #     pickle.dump(adjs, fp)
    if dataset == "email":
        adjs = adjs[0:20]
        # adjs = adjs
    view_network(adjs)
    return adjs


def load_tn_wiki(dataset):
    edge_file = open("data/{}.edge".format(dataset), 'r')
    edges = edge_file.readlines()
    node_num = int(edges[0].split('\t')[1].strip())
    edge_num = int(edges[1].split('\t')[1].strip())
    print("node number:{} , edge_number:{}".format(node_num, edge_num))
    edges.pop(0)
    edges.pop(0)
    start_year = 2004
    final_year = 2008
    period = final_year - start_year + 1
    shape = [period, node_num, node_num]
    adjs = np.zeros(shape)
    set_1 = set([])
    set_2 = set([])
    for line in edges:

        time = int(line.split('\t')[2].strip()) - start_year
        node1 = int(line.split('\t')[0].strip())
        node2 = int(line.split('\t')[1].strip())
        if time == 2005:
            set_1.add(node1)
            set_1.add(node2)
        if time == 2007:
            set_2.add(node1)
            set_2.add(node2)
        adjs[time][node1][node2] = 1.0
    # print(len(set_1))
    # print(len(set_2))
    # print(len(set_1 & set_2))

    return adjs


def load_tn(dataset):
    print("="*20)
    print("load dataset:{}".format(dataset))
    if dataset in ["email", "college"]:
        file = dataset + ".edge"
        with open(file, 'rb') as f:
            adjs = pickle.load(f)

    elif dataset in ["wiki"]:
        adjs = load_tn_wiki("wiki")
        adjs = adjs[1:4]
    elif dataset in ['email_20']:
        file = dataset + ".edge"
        with open(file, 'rb') as f:
            adjs = pickle.load(f)
    else:
        adjs = None
    print("load dataset:{} finished".format(dataset))
    print("time steps:{}".format(adjs.shape[0]))
    print("=" * 20)
    return adjs

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""

    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + np.eye(adj.shape[0]) + adj.T)
    return adj_normalized


def preprocess_adjs(adjs):
    adjs_normalized = []
    for adj in adjs:
        adjs_normalized.append(preprocess_adj(adj))
    return np.array(adjs_normalized)


# adjs = load_tn_wt("email")
# with open('email_20.edge', 'wb') as f:
#     pickle.dump(adjs, f)
# with open('email.edge', 'rb') as f:
#     adjs = pickle.load(f)
#     adjs = torch.from_numpy(adjs)




