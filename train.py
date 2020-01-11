import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import *
import os
import load_data
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import utils
import sys
import argparse
import time
from sklearn.utils import shuffle
import pickle


def train(args):
    # email 0,9 seed = 3
    adjs = load_data.load_tn("email")

    node_num = adjs.shape[1]
    step_num = 10  # adjs.shape[0]

    # x = torch.randn(10, node_num, node_num)
    x = torch.eye(node_num).unsqueeze(0)
    x = x.repeat(step_num, 1, 1)

    # create train and test
    time_1 = 0
    time_2 = 9
    seed = 3
    train_num = int(node_num // 10 * seed)
    test_num = node_num - train_num
    train = np.array([np.arange(0, train_num), np.arange(0, train_num)]).transpose()
    sample_nodes = utils.get_test(np.arange(0, node_num), adjs[time_1], adjs[time_2])
    test = utils.get_test(np.arange(train_num, node_num), adjs[time_1], adjs[time_2])
    test = np.array([test, test]).transpose()
    #     test = np.array([np.arange(train_num, node_num), np.arange(train_num, node_num)]).transpose()

    eye_train = torch.eye(train_num, node_num)
    features1_temp = torch.zeros(size=(test_num, node_num)).type(torch.float)
    features2_temp = torch.zeros(size=(test_num, node_num)).type(torch.float)
    features1 = torch.cat([eye_train, features1_temp], 0)
    features2 = torch.cat([eye_train, features2_temp], 0)

    adjs = load_data.preprocess_adjs(adjs)
    adjs = torch.from_numpy(adjs[0:step_num]).type(torch.float)
    x[time_1] = features1
    x[time_2] = features2

    # create model
    model = DGAM_RNN(node_num,
                 hidden_size=args.hidden_size,
                 emb_size=args.emb_size,
                 dropout=args.dropout,
                 num_layers=args.num_layers,
                 gamma=args.gamma,
                 num_sample=args.num_sample,
                 n_head=args.n_head,
                 dim_feedforward=args.dim_feedforward,
                 sample_nodes=sample_nodes)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    # train model
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        model.forward(x, adjs, train)
        loss_train = model.calculate_loss(train, time_1, time_2)
        # loss_test = model.evaluate(test)
        print("epoch:{} loss_train:{} ".format(epoch,
                                               loss_train.item(),
                                               ))
        loss_train.backward()
        optimizer.step()

    print("======" * 60)
    print("begin test")
    print("test_num:{}".format(test.shape[0]))
    embs = model.forward(x, adjs, train).data.numpy()
    embs1 = embs[time_1]
    embs2 = embs[time_2]
    utils.get_hits(embs1, embs2, test)
    print("finish test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='model parameters')
    parser.add_argument("--hidden-size", type=int, default=256,
                        help="hidden size of gcn")
    parser.add_argument("--emb-size", type=int, default=128,
                        help="embedding size")
    parser.add_argument('--dropout', type=float, default=0.,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--epochs', type=int, default=400,
                        help='Number of epochs to train.')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of layers.')
    parser.add_argument('--n_head', type=int, default=8,
                        help='Number of hedas.')
    parser.add_argument('--gamma', type=float, default=3.0,
                        help='Number of layers.')
    parser.add_argument('--dim_feedforward', type=int, default=1024,
                        help='Number of layers.')
    parser.add_argument('--num_sample', type=int, default=5,
                        help='Number of negative samples.')

    args = parser.parse_args([])
    print(args)
    train(args)

