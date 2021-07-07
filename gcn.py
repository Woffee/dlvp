import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

import time
from datetime import datetime

import networkx as nx
import numpy as np

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset, Dataset, download_url

import torch_geometric.transforms as T

from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import os

import logging


try:
   import cPickle as pickle
except:
   import pickle


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# log file
now_time = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(filename)s line: %(lineno)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=BASE_DIR + '/logs/' + now_time + '.log')
logger = logging.getLogger(__name__)


class GNNStack(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, lp_path_num, lp_length, lp_dim, ns_length, ns_dim, task='node'):
        """
        :param input_dim: max(dataset.num_node_features, 1)
        :param hidden_dim: 32
        :param output_dim: dataset.num_classes
        :param task: 'node' or 'graph'
        """
        super(GNNStack, self).__init__()
        self.task = task

        self.lp_path_num = lp_path_num
        self.lp_length = lp_length
        self.ns_length = ns_length


        # LP: (lp_path_num, lp_length, lp_dim)
        # self.lp_conv1 = nn.Conv2d(lp_path_num, 6, kernel_size=5, padding=2) # (6, lp_length, lp_dim)
        # self.lp_pool1 = nn.MaxPool2d(2, 2)
        # self.lp_conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2) # (16, lp_length / 2, lp_dim / 2)
        # self.lp_pool2 = nn.MaxPool2d(2, 2)
        # self.lp_fc = nn.Linear(16 * (lp_length/4) * (lp_dim/4), 128)
        self.lp_grus = nn.ModuleList()
        for l in range(lp_path_num):
            self.lp_grus.append(nn.GRU(128, lp_length, 1, batch_first=True))
        self.lp_fc = nn.Linear(128 * lp_path_num, 128)

        # NS: (1, ns_length, ns_dim)
        # self.ns_conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # (6, ns_length, ns_dim)
        # self.ns_pool1 = nn.MaxPool2d(2, 2)
        # self.ns_conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2)  # (16, ns_length / 2, ns_dim / 2)
        # self.ns_pool2 = nn.MaxPool2d(2, 2)
        # self.ns_fc = nn.Linear(16 * (ns_length / 4) * (ns_dim / 4), 128)
        self.ns_gru = nn.GRU(128, ns_length, 1, batch_first=True)

        # 5 个 embedding 合到一起
        self.all_fc = nn.Linear(128 * 5, 128)


        # Graph conv
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.lns = nn.ModuleList()
        self.lns.append(nn.LayerNorm(hidden_dim))
        self.lns.append(nn.LayerNorm(hidden_dim))
        for l in range(2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25),
            nn.Linear(hidden_dim, output_dim))
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.dropout = 0.25
        self.num_layers = 3 # len of self.convs

    def build_conv_model(self, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        if self.task == 'node':
            return pyg_nn.GCNConv(input_dim, hidden_dim)
        else:
            return pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                  nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))

    def forward(self, data):
        # x: [2265, 3]
        # edge_index: [2, 8468]
        # batch: 2265
        xs, edge_index, batch = data.xs, data.edge_index, data.batch
        # if data.num_node_features == 0:
        #   x = torch.ones(data.num_nodes, 1)
        logger.info("xs.length: %d" % len(xs) )

        # combine DEF, REF, PDT, LP, NS --> x
        # x_pdt = xs['pdt']
        # x_ref = xs['ref']
        # x_def = xs['def']
        # x_lp = xs['lp']
        # x_ns = xs['ns']

        x_pdt = []
        x_ref = []
        x_def = []
        x_lp = []
        x_ns = []
        for v_batch in xs:
            # print("vvv:", v)
            for v in v_batch:
                x_pdt.append(v['pdt'])
                x_ref.append(v['ref'])
                x_def.append(v['def'])
                x_lp.append(v['lp'])
                x_ns.append(v['ns'])
        x_pdt = torch.tensor(x_pdt, dtype=torch.float)
        x_ref = torch.tensor(x_ref, dtype=torch.float)
        x_def = torch.tensor(x_def, dtype=torch.float)
        x_lp = torch.tensor(x_lp, dtype=torch.float)
        x_ns = torch.tensor(x_ns, dtype=torch.float)

        # LP
        gru_output_list = []
        for i in range(self.lp_path_num):
            gru_output_list.append( self.lp_grus[i]( x_lp[i] ) )
        x_lp = torch.cat(gru_output_list)
        x_lp = self.lp_fc(x_lp)

        # NS
        x_ns = self.ns_gru(x_ns)

        # concatenate together
        x = self.all_fc(torch.cat( [x_pdt, x_ref, x_def, x_lp, x_ns]))

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            emb = x
            # emb: [2265, 32]
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        if self.task == 'graph':
            x = pyg_nn.global_mean_pool(x, batch)

        x = self.post_mp(x)

        return emb, F.log_softmax(x, dim=1) # dim (int): A dimension along which log_softmax will be computed.

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

class CustomConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CustomConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Add self-loops to the adjacency matrix.
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        # Transform node feature matrix.
        self_x = self.lin_self(x)
        #x = self.lin(x)

        return self_x + self.propagate(edge_index, size=(x.size(0), x.size(0)), x=self.lin(x))

    def message(self, x_i, x_j, edge_index, size):
        # Compute messages
        # x_j has shape [E, out_channels]

        row, col = edge_index
        deg = pyg_utils.degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        return aggr_out


def test(loader, model, is_validation=False):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():
            emb, pred = model(data)
            pred = pred.argmax(dim=1)
            label = data.y

        if model.task == 'node':
            mask = data.val_mask if is_validation else data.test_mask
            # node classification: only evaluate on nodes in test set
            pred = pred[mask]
            label = data.y[mask]

        correct += pred.eq(label).sum().item()

    if model.task == 'graph':
        total = len(loader.dataset)
    else:
        total = 0
        for data in loader.dataset:
            total += torch.sum(data.test_mask).item()
    return correct / total


def train(dataset, task, writer):
    if task == 'graph':
        data_size = len(dataset)
        print("data_size:", data_size)
        loader = DataLoader(dataset[:int(data_size * 0.8)], batch_size=64, shuffle=True)
        test_loader = DataLoader(dataset[int(data_size * 0.8):], batch_size=64, shuffle=True)
    else:
        test_loader = loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # lp_path_num, lp_length, lp_dim, ns_length, ns_dim,
    lp_length = dataset[0].xs[0]['lp'][0].shape[0]
    ns_length = dataset[0].xs[0]['ns'].shape[0]
    logger.info("== lp_length: %d" % lp_length)
    logger.info("== ns_length: %d" % ns_length)


    # build model
    model = GNNStack(input_dim=128,
                     hidden_dim=32,
                     output_dim=2,
                     lp_path_num = 20,
                     lp_length = lp_length,
                     lp_dim = 128,
                     ns_length = ns_length,
                     ns_dim=128,
                     task=task)
    print("len(model.convs):", len(model.convs))

    opt = optim.Adam(model.parameters(), lr=0.01)

    # train
    for epoch in range(800):
        total_loss = 0
        model.train()
        for batch in loader:
            # print(batch.train_mask, '----')
            opt.zero_grad()
            embedding, pred = model(batch)
            label = batch.y
            if task == 'node':
                pred = pred[batch.train_mask]
                label = label[batch.train_mask]
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        writer.add_scalar("loss", total_loss, epoch)

        if epoch % 10 == 0:
            test_acc = test(test_loader, model)
            print("Epoch {}. Loss: {:.4f}. Test accuracy: {:.4f}".format(
                epoch, total_loss, test_acc))
            logger.info("Epoch {}. Loss: {:.4f}. Test accuracy: {:.4f}".format(
                epoch, total_loss, test_acc))
            writer.add_scalar("test accuracy", test_acc, epoch)

    return model


def read_my_data(save_path):
    """
    :param save_path: '/xye_data_nobackup/wenbo/dlvp/data/function2vec'
    :return:
    """

    all_funcs_lp_file = save_path + "/all_functions_with_trees_lp.csv"

    emb_file_def = save_path + "/all_func_embedding_file_def.csv"
    emb_file_ref = save_path + "/all_func_embedding_file_ref.csv"
    emb_file_pdt = save_path + "/all_func_embedding_file_pdt.csv"

    emb_file_lp_combine = save_path + "/all_func_embedding_file_lp.pkl.combine"
    emb_file_lp_greedy = save_path + "/all_func_embedding_file_lp.pkl.greedy"

    emb_file_ns = save_path + "/all_func_embedding_file_ns.pkl"


    # read files
    df_all_funcs = pd.read_csv(all_funcs_lp_file)

    df_emb_def = pd.read_csv(emb_file_def).drop(columns=['type']).values
    df_emb_ref = pd.read_csv(emb_file_ref).drop(columns=['type']).values
    df_emb_pdt = pd.read_csv(emb_file_pdt).drop(columns=['type']).values

    emb_lp_combine = pickle.load(open(emb_file_lp_combine,"rb"))
    emb_lp_greedy  = pickle.load(open(emb_file_lp_greedy,"rb"))

    emb_ns = pickle.load(open(emb_file_ns, "rb"))

    print( "len(df_all_funcs):", len(df_all_funcs) )

    print( "len(df_emb_def):", len(df_emb_def) )
    print( "len(df_emb_ref):", len(df_emb_ref) )
    print( "len(df_emb_pdt):", len(df_emb_pdt) )

    print( "len(emb_lp_combine):", len(emb_lp_combine) )
    print( "len(emb_lp_greedy):", len(emb_lp_greedy) )

    print( "len(emb_ns):", len(emb_ns) )


    """
    len(df_all_funcs): 15719
    len(df_emb_def): 15719
    len(df_emb_ref): 15719
    len(df_emb_pdt): 15719
    len(emb_lp_combine): 14608
    len(emb_lp_greedy): 14608
    len(emb_ns): 15719
    """

    x = []
    edge_index = []
    y = []
    xs = []

    dataset = []
    for index, row in df_all_funcs.iterrows():
        func_id = int(row['func_id'])
        if not (func_id in emb_lp_combine.keys() and func_id in emb_ns.keys()):
            continue

        x.append([0])
        if row['callees'].strip() != "[]":
            logger.info("== callees != []")
            callees = row['callees'].strip().replace("[", "").replace("]","").replace(" ", "" ).split(",")
            logger.info("== len(callees):%d" % len(callees))
            for ce in callees:
                edge_index.append([ int(row['func_id']), int(ce) ])
        if row['callers'].strip() != "[]":
            callers = row['callers'].strip().replace("[", "").replace("]","").replace(" ", "" ).split(",")
            for cr in callers:
                edge_index.append([ int(cr), int(row['func_id']) ])

        xs.append({
            'lp': emb_lp_combine[func_id],
            'ns': emb_ns[func_id],
            'def': df_emb_def[index],
            'ref': df_emb_ref[index],
            'pdt': df_emb_pdt[index]
        })

        if int(row['is_center']) == 1:
            # y.append(int(row['vul']))
            # edge_index = np.array(edge_index)
            # edge_index = edge_index - edge_index.min()
            x = torch.tensor(x, dtype=torch.float)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            if edge_index.shape[0] > 0:
                edge_index = edge_index - edge_index.min()

            if int(row['vul']) == 1:
                y = torch.tensor([1, 0], dtype=torch.long)
            else:
                y = torch.tensor([0, 1], dtype=torch.long)

            data = Data(x=x, edge_index=edge_index.t().contiguous(), xs=xs, y=y)
            dataset.append(data)
            x = []
            edge_index = []
            y = []
            xs = []
    # loader = DataLoader(dataset, batch_size=2, shuffle=True)
    # return loader

    return dataset




if __name__ == '__main__':
    # save_path = '/xye_data_nobackup/wenbo/dlvp/data/function2vec'
    # save_path = BASE_DIR + "/data/function2vec"
    # read_my_data(save_path)
    # exit()


    # edge_index = torch.tensor([[10, 11],
    #                            [11, 10],
    #                            [11, 12],
    #                            [12, 11]], dtype=torch.long)
    # x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    # xs = [
    #     {
    #         'lp': torch.tensor([[-1], [0], [1]], dtype=torch.float)
    #     },
    #     {
    #         'lp': torch.tensor([[-1], [0], [1]], dtype=torch.float)
    #     },
    #     {
    #         'lp': torch.tensor([[-1], [0], [1]], dtype=torch.float)
    #     },
    # ]
    # data = Data(x=x, edge_index=edge_index.t().contiguous(), xs = xs)
    # loader = DataLoader([data], batch_size=2, shuffle=True)
    # for batch_data in loader:
    #     xs, edge_index, batch = batch_data.xs, batch_data.edge_index, batch_data.batch
    #     print("???", xs)
    #     print("===", edge_index)
    # exit()


    writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

    # dataset = TUDataset(root='/tmp/PROTEINS', name='PROTEINS')
    # dataset = dataset.shuffle()

    task = 'graph'
    save_path = BASE_DIR + "/data/function2vec"
    dataset = read_my_data(save_path)

    model = train(dataset, task, writer)