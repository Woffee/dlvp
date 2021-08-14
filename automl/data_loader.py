"""


@Time    : 8/9/21
@Author  : Wenbo
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from gensim.models.word2vec import Word2Vec

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

import time
from datetime import datetime

import networkx as nx

# from torch_geometric.datasets import TUDataset
# from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset, Dataset, download_url

# import torch_geometric.transforms as T

from tensorboardX import SummaryWriter
# from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import os

import logging
from pathlib import Path
import json
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
try:
   import cPickle as pickle
except:
   import pickle

import random
random.seed(9)



def findAllFile(base, full=False):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if full:
                yield os.path.join(root, f)
            else:
                yield f


def read_keys(keys_file):
    keys_index = {}
    ii = 0
    with open(keys_file, "r") as f:
        for line in f.readlines():
            l = line.strip()
            if l != "":
                keys_index[l] = ii
                ii += 1
    return keys_index

def get_1v1_graphs(graphs):
    res = []

    vuls = []
    non_vuls = []

    for g in graphs:
        y = g['vul']
        if y == 1:
            vuls.append( g )
        else:
            non_vuls.append( g )

    n = min( len(vuls), len(non_vuls) )
    logging.info("vuls: {}, non_vuls: {}, n: {}".format( len(vuls), len(non_vuls), n ))

    # 打乱顺序
    random.shuffle( vuls )
    random.shuffle( non_vuls )

    for i in range(n):
        res.append( vuls[i] )
        res.append( non_vuls[i] )
    return res


class MyLargeDataset(Dataset):
    def __init__(self, root="", params={},transform=None, pre_transform=None):
        self.save_path = root
        self.params = params
        super(MyLargeDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        res = []
        for f in findAllFile(self.processed_dir, full=True):
            if f.find("data_") >= 0 and f.endswith(".pt"):
                pos = f.find("processed") + 10
                res.append(f[pos:])
        return res

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        params = self.params

        emb_file_def = params['embedding_path'] + "/all_func_embedding_def.csv"
        emb_file_def_keys = params['embedding_path'] + "/all_func_embedding_def_keys.txt"
        emb_file_ref = params['embedding_path'] + "/all_func_embedding_ref.csv"
        emb_file_ref_keys = params['embedding_path'] + "/all_func_embedding_ref_keys.txt"
        emb_file_pdt = params['embedding_path'] + "/all_func_embedding_pdt.csv"
        emb_file_pdt_keys = params['embedding_path'] + "/all_func_embedding_pdt_keys.txt"

        emb_file_ns = params['embedding_path'] + "/all_func_embedding_ns.pkl"

        # read files
        df_emb_def = pd.read_csv(emb_file_def).drop(columns=['type']).values
        keys_index_def = read_keys(emb_file_def_keys)
        df_emb_ref = pd.read_csv(emb_file_ref).drop(columns=['type']).values
        keys_index_ref = read_keys(emb_file_ref_keys)
        df_emb_pdt = pd.read_csv(emb_file_pdt).drop(columns=['type']).values
        keys_index_pdt = read_keys(emb_file_pdt_keys)

        emb_ns = pickle.load(open(emb_file_ns, "rb"))

        print("len(df_emb_def):", len(df_emb_def))
        print("len(df_emb_ref):", len(df_emb_ref))
        print("len(df_emb_pdt):", len(df_emb_pdt))

        # print("len(emb_lp_combine):", len(emb_lp_combine))
        # print("len(emb_lp_greedy):", len(emb_lp_greedy))
        print("len(emb_ns):", len(emb_ns))


        graphs_file = params['embedding_path'] + "/graphs.pkl"
        logging.info("loading graphs from: {}".format(graphs_file))
        with open(graphs_file, 'rb') as fh:
            graphs = pickle.load(fh)

        graphs = get_1v1_graphs(graphs)

        ii = 0
        for g in graphs:
            func_key = g['func_key']
            y = int(g['vul'])

            if func_key not in keys_index_pdt.keys():
                continue
            if func_key not in keys_index_ref.keys():
                continue
            if func_key not in keys_index_def.keys():
                continue

            x_pdt = torch.tensor(df_emb_pdt[keys_index_pdt[func_key]], dtype=torch.float)
            x_ref = torch.tensor(df_emb_ref[keys_index_ref[func_key]], dtype=torch.float)
            x_def = torch.tensor(df_emb_def[keys_index_def[func_key]], dtype=torch.float)

            data = Data(num_nodes=1,
                        y=y,
                        x_pdt=x_pdt,
                        x_ref=x_ref,
                        x_def=x_def,
                        )

            save_path = os.path.join(self.processed_dir, str(ii // 1000))
            Path(save_path).mkdir(parents=True, exist_ok=True)

            to_file = os.path.join(save_path, 'data_{}.pt'.format(ii))
            torch.save(data, to_file)

            ii += 1

            logging.info("saved to: {}, vul: {}".format(to_file, y))


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        save_path = os.path.join(self.processed_dir, str(idx // 1000))
        data = torch.load(os.path.join(save_path, 'data_{}.pt'.format(idx)))
        return data
