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

LP_PATH_NUM = 20
BATCH_SIZE = 64

class GNNStack(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, lp_path_num, lp_length, lp_dim, ns_length, ns_dim,
                 task='node'):
        """
        :param input_dim: max(dataset.num_node_features, 1)
        :param hidden_dim: 128
        :param output_dim: dataset.num_classes
        :param task: 'node' or 'graph'
        """
        super(GNNStack, self).__init__()
        self.task = task

        self.lp_path_num = lp_path_num
        self.lp_length = lp_length
        self.ns_length = ns_length

        self.lp_grus = nn.ModuleList()
        for l in range(lp_path_num):
            """
            input_size – The number of expected features in the input x
            hidden_size – The number of features in the hidden state h
            num_layers – Number of recurrent layers. 
            """
            self.lp_grus.append(nn.GRU(input_size=lp_dim, hidden_size=lp_dim, num_layers=1, batch_first=True))
        self.lp_fc = nn.Linear(lp_dim * lp_path_num, 128)

        self.ns_gru = nn.GRU(lp_dim, lp_dim, 1, batch_first=True)

        # 5 个 embedding 合到一起
        self.all_fc = nn.Linear(lp_dim * 5, 128)


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
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # if data.num_node_features == 0:
        #   x = torch.ones(data.num_nodes, 1)
        # logger.info("xs.length: %d" % len(xs) )

        # DEF, REF, PDT, LP, NS --> x
        x_lp, x_ns, x_ref, x_def, x_pdt = data.x_lp, data.x_ns, data.x_ref, data.x_def, data.x_pdt
        # print("x_ns:", x_ns.shape)
        # print("x_lp:", x_lp.shape)

        # 当前 batch 总共有多少个 function。因为每个 graph 含有不同数量的 function，因此这里数值不定。
        cur_nodes_size = x_ns.shape[0]

        x_lp_list = torch.zeros([LP_PATH_NUM, x_ns.shape[0], 60, 128], dtype=torch.float)
        for i in range(LP_PATH_NUM):
            for j in range(x_ns.shape[0]):
                x_lp_list[i][j] = x_lp[j][i]

        # LP
        gru_output_list = []
        for i in range(self.lp_path_num):
            out, h = self.lp_grus[i](x_lp_list[i])
            # print("h:", h.shape) # h: torch.Size([1, 64, 128])
            gru_output_list.append(h)
        # gru_output_list = torch.tensor(gru_output_list, dtype=torch.float)
        # gru_output_list = gru_output_list.view(64, 128 * 20)
        x_lp = torch.cat(gru_output_list).view(x_ns.shape[0], 128 * 20)
        x_lp = self.lp_fc(x_lp)
        # print("x_lp_after:", x_lp.shape)

        # NS
        out, h_ns = self.ns_gru(x_ns)
        # print("h_ns:", h_ns.shape)
        x_ns = h_ns.view(cur_nodes_size, 128)

        # concatenate together
        x = torch.cat([x_pdt, x_ref, x_def, x_lp, x_ns]).view(cur_nodes_size, 128 * 5)
        x = self.all_fc(x)

        # print("x_concatenated:", x.shape)
        # print("edge_index:", edge_index.shape)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            emb = x
            # emb: [2265, 32]
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)
        # print("x after convs:", x.shape)
        if self.task == 'graph':
            x = pyg_nn.global_mean_pool(x, batch)
        # print("batch:", batch)
        # print("x after global_mean_pool:", x.shape)

        x = self.post_mp(x)
        # print("x_final.shape:", x.shape)
        return emb, F.log_softmax(x, dim=1)  # dim (int): A dimension along which log_softmax will be computed.

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
    lp_length = dataset[0].x_lp[0][0].shape[0]
    ns_length = dataset[0].x_ns[0].shape[0]
    logger.info("== lp_length: %d" % lp_length)
    logger.info("== ns_length: %d" % ns_length)
    print("== lp_length: %d" % lp_length)
    print("== ns_length: %d" % ns_length)


    # build model
    model = GNNStack(input_dim=128,
                     hidden_dim=32,
                     output_dim=2,
                     lp_path_num = LP_PATH_NUM,
                     lp_length = lp_length,
                     lp_dim = 128,
                     ns_length = ns_length,
                     ns_dim=128,
                     task=task)
    print("len(model.convs):", len(model.convs))

    opt = optim.Adam(model.parameters(), lr=0.01)

    # train
    for epoch in range(800):
        logger.info("=== now epoch: %d" % epoch)
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
            # print("pred.shape : ",pred.shape)
            # print("label.shape: ",label.shape)

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

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            yield f


def get_graph_data(df, deleted, func_id, func_index, nodes, edge_index):
    """
    nodes : [func_id, func_id ...]
    edges : [ [index_of_func_id, index_of_func_id]... ]
    func_index: func_id 在 nodes 中的 index
    """
    # print("get_graph_data - func_id:", func_id)
    row = df.iloc[func_id]
    if row['callees'].strip() != "[]":
        callees = row['callees'].strip().replace("[", "").replace("]", "").replace(" ", "").split(",")
        for ce in callees:
            ce_func_id = int(ce)
            if ce_func_id in deleted:
                continue

            if ce_func_id not in nodes:
                nodes.append(ce_func_id)
                ce_index = len(nodes) - 1
            else:
                ce_index = nodes.index(ce_func_id)

            edge_index.append([func_index, ce_index])
            nodes, edge_index = get_graph_data(df, deleted, ce_func_id, ce_index, nodes, edge_index)

    if row['callers'].strip() != "[]":
        callers = row['callers'].strip().replace("[", "").replace("]", "").replace(" ", "").split(",")
        for cr in callers:
            cr_func_id = int(cr)
            if cr_func_id in deleted:
                continue

            if cr_func_id not in nodes:
                nodes.append(cr_func_id)
                cr_index = len(nodes) - 1
            else:
                cr_index = nodes.index(cr_func_id)

            edge_index.append([cr_index, func_index])
            nodes, edge_index = get_graph_data(df, deleted, int(cr), cr_index, nodes, edge_index)

    return nodes, edge_index


class MyLargeDataset(Dataset):
    def __init__(self, root="", transform=None, pre_transform=None):
        self.save_path = root
        super(MyLargeDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        res = []
        # print("self.processed_dir:", self.processed_dir)
        for f in findAllFile(self.processed_dir):
            # print("file: ", f)
            if f.startswith("data_") and f.endswith(".pt"):
                res.append(f)

        print("processed_file_names:", res)
        return res

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):

        all_funcs_lp_file = self.save_path + "/all_functions_with_trees_lp.csv"

        emb_file_def = self.save_path + "/all_func_embedding_file_def.csv"
        emb_file_ref = self.save_path + "/all_func_embedding_file_ref.csv"
        emb_file_pdt = self.save_path + "/all_func_embedding_file_pdt.csv"

        emb_file_lp_combine = self.save_path + "/all_func_embedding_file_lp.pkl.combine"
        emb_file_lp_greedy = self.save_path + "/all_func_embedding_file_lp.pkl.greedy"

        emb_file_ns = self.save_path + "/all_func_embedding_file_ns.pkl"

        # read files
        df_all_funcs = pd.read_csv(all_funcs_lp_file)

        df_emb_def = pd.read_csv(emb_file_def).drop(columns=['type']).values
        df_emb_ref = pd.read_csv(emb_file_ref).drop(columns=['type']).values
        df_emb_pdt = pd.read_csv(emb_file_pdt).drop(columns=['type']).values

        emb_lp_combine = pickle.load(open(emb_file_lp_combine, "rb"))
        emb_lp_greedy = pickle.load(open(emb_file_lp_greedy, "rb"))

        emb_ns = pickle.load(open(emb_file_ns, "rb"))

        print("len(df_all_funcs):", len(df_all_funcs))

        print("len(df_emb_def):", len(df_emb_def))
        print("len(df_emb_ref):", len(df_emb_ref))
        print("len(df_emb_pdt):", len(df_emb_pdt))

        print("len(emb_lp_combine):", len(emb_lp_combine))
        print("len(emb_lp_greedy):", len(emb_lp_greedy))
        print("len(emb_ns):", len(emb_ns))

        df = df_all_funcs

        # 有些 function 无法生成 LP 和 NS，抛弃掉
        deleted = []
        for index, row in df.iterrows():
            func_id = int(row['func_id'])
            if not (func_id in emb_lp_combine.keys() and func_id in emb_ns.keys()):
                deleted.append( func_id )

        ii = 0
        for index, row in df.iterrows():
            func_id = int(row['func_id'])
            if func_id > 1000:
                break
            if func_id in deleted:
                continue

            if int(row['is_center']) == 1:
                nodes = [func_id]
                func_index = 0
                edge_index = []

                nodes, edge_index = get_graph_data(df, deleted, func_id, func_index, nodes, edge_index)
                # print("nodes:\n", nodes)
                # print("edge_index:\n", edge_index)
                if len(edge_index) == 0:
                    continue

                nodes_num = len(nodes)
                edges_num = len(edge_index)

                x_lp = []
                x_ns = []
                x_def = []
                x_ref = []
                x_pdt = []
                for f_index, f_id in enumerate(nodes):

                    x_lp_emb = np.zeros((LP_PATH_NUM, 60, 128), dtype=float)

                    for i in range(len(emb_lp_combine[f_id])):
                        if i >= LP_PATH_NUM:
                            break
                        x_lp_emb[i] = torch.tensor(emb_lp_combine[f_id][i], dtype=torch.float)

                    x_lp.append(x_lp_emb)
                    x_ns.append(emb_ns[f_id])
                    x_def.append(df_emb_def[index])
                    x_ref.append(df_emb_ref[index])
                    x_pdt.append(df_emb_pdt[index])

                x_lp = torch.tensor(np.array(x_lp), dtype=torch.float)
                x_ns = torch.tensor(x_ns, dtype=torch.float)
                x_def = torch.tensor(x_def, dtype=torch.float)
                x_ref = torch.tensor(x_ref, dtype=torch.float)
                x_pdt = torch.tensor(x_pdt, dtype=torch.float)

                edge_index = torch.tensor(edge_index, dtype=torch.long)
                if edge_index.shape[0] > 0:
                    edge_index = edge_index - edge_index.min()

                if int(row['vul']) == 1:
                    y = 1
                else:
                    y = 0

                data = Data(num_nodes=x_ns.shape[0],
                            edge_index=edge_index.t().contiguous(),
                            y=y,
                            x_def=x_def,
                            x_ref=x_ref,
                            x_pdt=x_pdt,
                            x_lp=x_lp,
                            x_ns=x_ns,
                            )

                # if self.pre_filter is not None and not self.pre_filter(data):
                #    continue

                # if self.pre_transform is not None:
                #    data = self.pre_transform(data)

                to_file = os.path.join(self.processed_dir, 'data_{}.pt'.format(ii))
                torch.save(data, to_file)
                print("saved to: %s, nodes_num: %d, edges_num: %d" % (to_file, nodes_num, edges_num))
                logger.info("saved to: %s, nodes_num: %d, edges_num: %d" % (to_file, nodes_num, edges_num))
                ii += 1


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data

if __name__ == '__main__':
    writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

    task = 'graph'
    save_path = BASE_DIR + "/data/function2vec"

    dataset = MyLargeDataset(save_path)
    model = train(dataset, task, writer)
    logger.info("training GCN done.")