"""


@Time    : 8/14/21
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

class GNNStack(nn.Module):

    def __init__(self, params):
        """
        :param input_dim: max(dataset.num_node_features, 1)
        :param hidden_dim: 128
        :param output_dim: dataset.num_classes
        :param task: 'node' or 'graph'
        """
        super(GNNStack, self).__init__()
        self.params = params

        self.input_dim = params['input_dim']
        self.hidden_dim = params['hidden_dim']
        self.output_dim = params['output_dim']
        self.batch_size = params['batch_size']

        self.fc1 = nn.Linear(params['input_dim']  * 3, params['input_dim']  * 2)
        self.fc2 = nn.Linear(params['input_dim']  * 2, params['hidden_dim'] )
        self.fc3 = nn.Linear(params['hidden_dim'] , params['output_dim'] )

        self.num_layers = 3
        self.loss_layer = nn.CrossEntropyLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, data):
        data.to(self.device)

        # DEF, REF, PDT, LP, NS --> x
        # idx_lp, idx_ns, x_ref, x_def, x_pdt = data.x_lp, data.x_ns, data.x_ref, data.x_def, data.x_pdt

        x_pdt, x_def, x_ref , batch = data.x_pdt, data.x_def, data.x_ref, data.batch

        x_pdt = x_pdt.view( -1, self.input_dim )
        x_def = x_def.view( -1, self.input_dim )
        x_ref = x_ref.view( -1, self.input_dim )

        x = torch.cat([x_pdt, x_ref, x_def], 1).to(self.device)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def loss(self, pred, label):
        # CrossEntropyLoss torch.nn.CrossEntropyLoss
        # return F.nll_loss(pred, label)
        # logger.info("pred: {}, label: {}".format(pred.shape, label.shape) )
        # exit()
        return self.loss_layer(pred, label)



class GNNStack2(nn.Module):

    def __init__(self, params):
        """
        :param input_dim: max(dataset.num_node_features, 1)
        :param hidden_dim: 128
        :param output_dim: dataset.num_classes
        :param task: 'node' or 'graph'
        """
        super(GNNStack2, self).__init__()
        self.params = params

        self.input_dim = params['input_dim']
        self.hidden_dim = params['hidden_dim']
        self.output_dim = params['output_dim']
        self.batch_size = params['batch_size']

        self.fc1 = nn.Linear(params['input_dim'] , 64 )
        self.fc2 = nn.Linear(64 * 3, params['input_dim'] )
        self.fc3 = nn.Linear(params['input_dim'], 64 )
        self.fc4 = nn.Linear(64, params['output_dim'] )

        self.num_layers = 3
        self.loss_layer = nn.CrossEntropyLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, data):
        data.to(self.device)

        # DEF, REF, PDT, LP, NS --> x
        # idx_lp, idx_ns, x_ref, x_def, x_pdt = data.x_lp, data.x_ns, data.x_ref, data.x_def, data.x_pdt

        x_pdt, x_def, x_ref , batch = data.x_pdt, data.x_def, data.x_ref, data.batch

        x_pdt = x_pdt.view( -1, self.input_dim )
        x_def = x_def.view( -1, self.input_dim )
        x_ref = x_ref.view( -1, self.input_dim )

        x_pdt = F.relu(self.fc1(x_pdt))
        x_ref = F.relu(self.fc1(x_ref))
        x_def = F.relu(self.fc1(x_def))

        x = torch.cat([x_pdt, x_ref, x_def], 1).to(self.device)

        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def loss(self, pred, label):
        # CrossEntropyLoss torch.nn.CrossEntropyLoss
        # return F.nll_loss(pred, label)
        # logger.info("pred: {}, label: {}".format(pred.shape, label.shape) )
        # exit()
        return self.loss_layer(pred, label)


class GNNStack3(nn.Module):

    def __init__(self, params):
        """
        :param input_dim: max(dataset.num_node_features, 1)
        :param hidden_dim: 128
        :param output_dim: dataset.num_classes
        :param task: 'node' or 'graph'
        """
        super(GNNStack3, self).__init__()
        self.params = params

        self.input_dim = params['input_dim']
        self.hidden_dim = params['hidden_dim']
        self.output_dim = params['output_dim']
        self.batch_size = params['batch_size']

        self.fc1 = nn.Linear(params['input_dim'] , params['hidden_dim'] )
        self.fc2 = nn.Linear(params['hidden_dim'] , params['output_dim'] )

        self.num_layers = 3
        self.loss_layer = nn.CrossEntropyLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, data):
        data.to(self.device)

        # DEF, REF, PDT, LP, NS --> x
        # idx_lp, idx_ns, x_ref, x_def, x_pdt = data.x_lp, data.x_ns, data.x_ref, data.x_def, data.x_pdt

        x_pdt, x_def, x_ref , batch = data.x_pdt, data.x_def, data.x_ref, data.batch

        x_pdt = x_pdt.view( -1, self.input_dim )
        x_def = x_def.view( -1, self.input_dim )
        x_ref = x_ref.view( -1, self.input_dim )

        x = x_pdt * x_def * x_ref

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def loss(self, pred, label):
        # CrossEntropyLoss torch.nn.CrossEntropyLoss
        # return F.nll_loss(pred, label)
        # logger.info("pred: {}, label: {}".format(pred.shape, label.shape) )
        # exit()
        return self.loss_layer(pred, label)