"""


@Time    : 8/9/21
@Author  : Wenbo
"""


import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
#
# from torch.nn.utils.rnn import pad_sequence
# from torch.nn.utils.rnn import pack_padded_sequence
# from torch.nn.utils.rnn import pad_packed_sequence
# from gensim.models.word2vec import Word2Vec
#
# import torch_geometric.nn as pyg_nn
# import torch_geometric.utils as pyg_utils

import time
from datetime import datetime

import networkx as nx
import numpy as np

from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarGraph
from gensim.models import Word2Vec

import matplotlib.pyplot as plt
import pandas as pd
import os

import logging
from pathlib import Path
import json
import argparse

try:
    import cPickle as pickle
except:
    import pickle



class Node2vec():
    def __init__(self, save_path):
        # self.save_path = save_path + "/"
        self.model_path = save_path + "/node2vec.bin"
        self.model = None
        self.embedding_file = save_path + "/node2vec_embeddings.pkl"

    def train(self, graphs):
        walk_length = 50
        weighted_walks = []
        for g in graphs:
            if g['G'] is not None:
                G = g['G']
                G = StellarGraph.from_networkx(G)
                # print(G.info())
                rw = BiasedRandomWalk(G)
                weighted_walks = weighted_walks + rw.run(
                    nodes=G.nodes(),  # root nodes
                    length=walk_length,  # maximum length of a random walk
                    n=1,    # number of random walks per root node
                    p=0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
                    q=2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
                    weighted=True,  # for weighted random walks
                    seed=42,  # random seed fixed for reproducibility
                )
                # print("Number of random walks: {}".format(len(weighted_walks)))

        logging.info("Number of random walks: {}".format(len(weighted_walks)))
        logging.info("training node2vec...")
        weighted_model = Word2Vec(
            weighted_walks, size=128, window=5, min_count=0, sg=1, workers=1, iter=1
        )

        weighted_model.save(self.model_path)
        self.model = weighted_model
        logging.info("training node2vec done, saved to: {}".format(self.model_path))

    def predict(self):
        pass

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = Word2Vec.load(self.model_path)


    def get_embedding(self, func_key):
        if self.model is None:
            self.load_model()

        if func_key in self.model.wv:
            return self.model.wv[func_key]
        else:
            logging.info("node2vec - no such a key: {}".format(func_key))
        return None

    def save_embeddings(self, graphs):
        embeddings = {}
        for g in graphs:
            func_key = g['func_key']
            embeddings[func_key] = self.get_embedding(func_key)

        pickle.dump(embeddings, open(self.embedding_file, "wb"))
        logging.info("len(n2v_embeddings): {}, saved to {}".format(len(embeddings), self.embedding_file))

    def load_embeddings(self):
        if os.path.exists(self.embedding_file):
            return pickle.load( open( self.embedding_file, "rb" ) )
