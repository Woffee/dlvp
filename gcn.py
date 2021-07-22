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
import numpy as np

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
from gpu_mem_track import MemTracker

import logging
from pathlib import Path
import json
import argparse

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

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('# Using device:', device)
logger.info("=== using device: %s" % str(device))

#Additional Info when using cuda
if device.type == 'cuda':
    print('#', torch.cuda.get_device_name(0))
    # print('# Memory Usage:')
    # print('# Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    # print('# Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

# args
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--tasks_file', help='tasks_file', type=str, default='cflow/tasks.json')
parser.add_argument('--functions_path', help='functions_path', type=str, default=BASE_DIR + "/data/function2vec2/functions_jy")
parser.add_argument('--model_save_path', help='model_save_path', type=str, default=BASE_DIR + "/data/gcn_models")

parser.add_argument('--input_dim', help='input_dim', type=int, default=128)
parser.add_argument('--hidden_dim', help='hidden_dim', type=int, default=128)
parser.add_argument('--batch_size', help='hidden_dim', type=int, default=16)

parser.add_argument('--lp_path_num', help='lp_path_num', type=int, default=20)
parser.add_argument('--lp_length', help='lp_length', type=int, default=60)
parser.add_argument('--lp_dim', help='lp_dim', type=int, default=128)
parser.add_argument('--lp_w2v_path', help='lp_w2v_path', type=str, default=BASE_DIR + "/data/function2vec2/models/w2v_lp_combine.bin")

parser.add_argument('--ns_length', help='ns_length', type=int, default=2000)
parser.add_argument('--ns_dim', help='ns_dim', type=int, default=128)
parser.add_argument('--ns_w2v_path', help='ns_w2v_path', type=str, default=BASE_DIR + "/data/function2vec2/models/w2v_ns.bin")
args = parser.parse_args()

MODEL_SAVE_PATH = args.model_save_path
Path(MODEL_SAVE_PATH).mkdir(parents=True, exist_ok=True)

FUNCTIONS_PATH = args.functions_path
TASKS_FILE = args.tasks_file

INPUT_DIM = args.input_dim
HIDDEN_DIM = args.hidden_dim
BATCH_SIZE = args.batch_size

# 每一个 function 的 LP 可表示为 (LP_PATH_NUM, LP_LENGTH, LP_DIM)
LP_PATH_NUM = args.lp_path_num
LP_LENGTH = args.lp_length
LP_DIM = args.lp_dim
LP_W2V_PATH = args.lp_w2v_path

# NS
NS_LENGTH = args.ns_length
NS_DIM = args.ns_dim
NS_W2V_PATH = args.ns_w2v_path

def plot_dataset(dataset):
    edges_raw = dataset.data.edge_index.numpy()
    edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]
    labels = dataset.data.y.numpy()

    G = nx.Graph()
    G.add_nodes_from(list(range(np.max(edges_raw))))
    G.add_edges_from(edges)
    plt.subplot(111)
    options = {
                'node_size': 30,
                'width': 0.2,
    }
    nx.draw(G, with_labels=False, node_color=labels.tolist(), cmap=plt.cm.tab10, font_weight='bold', **options)
    plt.show()


def myprint(s, is_log=0):
    print(s)
    if is_log>0:
        logger.info(s)

class GNNStack(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, lp_path_num, lp_length, lp_dim, lp_w2v_path, ns_length, ns_dim, ns_w2v_path,
                 task='node'):
        """
        :param input_dim: max(dataset.num_node_features, 1)
        :param hidden_dim: 128
        :param output_dim: dataset.num_classes
        :param task: 'node' or 'graph'
        """
        super(GNNStack, self).__init__()
        self.task = task

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.lp_path_num = lp_path_num
        self.lp_length = lp_length
        self.lp_dim = lp_dim

        self.ns_length = ns_length
        self.ns_dim = ns_dim

        self.lp_w2v_model = Word2Vec.load(lp_w2v_path)
        self.lp_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(self.lp_w2v_model.wv.vectors))
        self.lp_gru = nn.GRU(input_size=lp_dim, hidden_size=lp_dim, num_layers=1, batch_first=True)
        self.lp_fc = nn.Linear(lp_dim * lp_path_num, input_dim)

        self.ns_w2v_model = Word2Vec.load(ns_w2v_path)
        self.ns_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(self.ns_w2v_model.wv.vectors))
        self.ns_gru = nn.GRU(ns_dim, lp_dim, 1, batch_first=True)

        # concatenate 5 embeddings
        self.all_fc = nn.Linear(lp_dim * 5, input_dim)


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
        self.loss_layer = nn.CrossEntropyLoss()

    def build_conv_model(self, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        if self.task == 'node':
            return pyg_nn.GCNConv(input_dim, hidden_dim)
        else:
            return pyg_nn.GCNConv(input_dim, hidden_dim)
            # return pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))

    def forward(self, data):
        data.to(device)

        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch

        # DEF, REF, PDT, LP, NS --> x
        idx_lp, idx_ns, x_ref, x_def, x_pdt = data.x_lp, data.x_ns, data.x_ref, data.x_def, data.x_pdt
        x_ns_length = data.x_ns_length

        # 当前 batch 总共有多少个 function。因为每个 graph 含有不同数量的 function，因此这里数值不定。
        cur_nodes_size = idx_ns.shape[0]

        # LP
        x_lp_list = [ [ torch.zeros(self.lp_length, dtype=torch.float, device=device) for j in range(cur_nodes_size) ] for i in range(self.lp_path_num) ]
        for i in range(self.lp_path_num):
            for j in range(idx_ns.shape[0]):
                x_lp_list[i][j] = idx_lp[j][i]
        x_lp_length = torch.ones([self.lp_path_num, cur_nodes_size], dtype=torch.long, device=device)
        for i in range(self.lp_path_num):
            for j in range(idx_ns.shape[0]):
                x_lp_length[i][j] = max(data.x_lp_length[j][i], 1)

        gru_output_list = []
        for i in range(self.lp_path_num):
            x_lp = pad_sequence(x_lp_list[i], batch_first=True, padding_value=0)
            x_lp = self.lp_embedding(x_lp)
            x_lp = pack_padded_sequence(x_lp, x_lp_length[i], batch_first=True, enforce_sorted=False)
            out, h = self.lp_gru(x_lp)
            # print("h:", h.shape) # h: torch.Size([1, 64, 128])
            gru_output_list.append(h)

        del x_lp_list
        x_lp = torch.cat(gru_output_list).view(cur_nodes_size, self.lp_dim * self.lp_path_num).to(device)
        del gru_output_list
        x_lp = self.lp_fc(x_lp)


        # NS
        idx_ns_list = [torch.zeros(self.ns_length, dtype=torch.float, device=device) for j in range(cur_nodes_size)]
        for i in range(cur_nodes_size):
            idx_ns_list[i] = idx_ns[i]
        x_ns = pad_sequence(idx_ns_list, batch_first=True, padding_value=0)
        x_ns = self.ns_embedding(x_ns)
        x_ns = pack_padded_sequence(x_ns, x_ns_length, batch_first=True, enforce_sorted=False)
        out, h_ns = self.ns_gru(x_ns)
        del out
        # print("h_ns:", h_ns.shape)
        x_ns = h_ns.view(cur_nodes_size, self.ns_dim)
        del h_ns

        # concatenate together
        x = torch.cat([x_pdt, x_ref, x_def, x_lp, x_ns]).view(cur_nodes_size, self.ns_dim * 5).to(device)
        del x_lp, x_ns, x_ref, x_def, x_pdt
        x = self.all_fc(x)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            emb = x
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        if self.task == 'graph':
            x = pyg_nn.global_mean_pool(x, batch)

        x = self.post_mp(x)
        return emb, F.log_softmax(x, dim=1)  # dim (int): A dimension along which log_softmax will be computed.

    def loss(self, pred, label):
        # CrossEntropyLoss torch.nn.CrossEntropyLoss
        # return F.nll_loss(pred, label)
        return self.loss_layer(pred, label)

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


def train(dataset, task, writer, plot=False):
    if task == 'graph':
        data_size = len(dataset)
        print("data_size:", data_size)
        loader = DataLoader(dataset[:int(data_size * 0.8)], batch_size=BATCH_SIZE, shuffle=True)
        vali_loader = DataLoader(dataset[int(data_size * 0.8): int(data_size * 0.9)], batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(dataset[int(data_size * 0.9):], batch_size=BATCH_SIZE, shuffle=True)
    else:
        test_loader = loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # lp_path_num, lp_length, lp_dim, ns_length, ns_dim,
    # lp_length = dataset[0].x_lp[0][0].shape[0]
    # ns_length = dataset[0].x_ns[0].shape[0]
    # logger.info("== lp_length: %d" % lp_length)
    # logger.info("== ns_length: %d" % ns_length)
    # print("== lp_length: %d" % lp_length)
    # print("== ns_length: %d" % ns_length)


    # build model
    model = GNNStack(input_dim=INPUT_DIM,
                     hidden_dim=HIDDEN_DIM,
                     output_dim=2,
                     lp_path_num = LP_PATH_NUM,
                     lp_length = LP_LENGTH,
                     lp_dim = LP_DIM,
                     lp_w2v_path = LP_W2V_PATH,
                     ns_length = NS_LENGTH,
                     ns_dim=NS_DIM,
                     ns_w2v_path= NS_W2V_PATH,
                     task=task)

    # gpu_tracker = MemTracker()  # define a GPU tracker
    # gpu_tracker.track()

    model.to(device)

    # gpu_tracker.track()

    print("len(model.convs):", len(model.convs))

    opt = optim.Adam(model.parameters(), lr=0.01)
    min_valid_loss = np.inf
    best_model_path = ""

    train_accuracies, vali_accuracies = list(), list()

    # train
    for epoch in range(200):
        logger.info("=== now epoch: %d" % epoch)
        total_loss = 0
        model.train()
        for batch in loader:
            # print(batch.train_mask, '----')
            opt.zero_grad() # 清空梯度
            embedding, pred = model(batch)
            label = batch.y.to(device)
            if task == 'node':
                pred = pred[batch.train_mask]
                label = label[batch.train_mask]
            # print("pred.shape : ",pred.shape)
            # print("label.shape: ",label.shape)

            loss = model.loss(pred, label)

            # gpu_tracker.track()

            # logger.info("# Allocated loss: %.3f GB" % round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1))
            # logger.info("# Cached    loss: %.3f GB" % round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1))

            loss.backward() # 反向计算梯度，累加到之前梯度上
            opt.step() # 更新参数
            total_loss += loss.item() * batch.num_graphs

            # gpu_tracker.track()

            # logger.info("# Allocated backward: %.3f GB" % round(torch.cuda.memory_allocated(0)/1024**3,1))
            # logger.info("# Cached    backward: %.3f GB" % round(torch.cuda.memory_reserved(0)/1024**3,1))

            # delete caches
            del embedding, pred, loss
            torch.cuda.empty_cache()

            # gpu_tracker.track()

            # logger.info("# Allocated empty_cache: %.3f GB" % round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1))
            # logger.info("# Cached    empty_cache: %.3f GB" % round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1))


        total_loss /= len(loader.dataset)
        writer.add_scalar("loss", total_loss, epoch)

        # validate
        train_acc = test(loader, model)
        vali_acc = test(vali_loader, model)

        train_accuracies.append(train_acc)
        vali_accuracies.append(vali_acc)

        if min_valid_loss > total_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{total_loss:.6f}) \t Saving The Model')
            logger.info(f'Validation Loss Decreased({min_valid_loss:.6f}--->{total_loss:.6f}) \t Saving The Model')
            min_valid_loss = total_loss
            # Saving State Dict
            torch.save(model.state_dict(), MODEL_SAVE_PATH + "/gcn_model_epoch%d.pth" % epoch)
            best_model_path = MODEL_SAVE_PATH + "/gcn_model_epoch%d.pth" % epoch

        myprint( "Epoch {}. Loss: {:.4f}. train_acc: {:.4f}. test_acc: {:.4f}".format(
            epoch, total_loss, train_acc, vali_acc), 1 )

        writer.add_scalar("test accuracy", vali_acc, epoch)

    # Load
    # model = Net()
    if os.path.exists(best_model_path):
        myprint("loading the best model: %s" % best_model_path, 1)
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

    if plot:
        plt.plot(train_accuracies, label="Train accuracy")
        plt.plot(vali_accuracies, label="Validation accuracy")
        plt.xlabel("# Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc='upper right')
        # plt.show()
        plt.savefig(MODEL_SAVE_PATH + "/gcn_model_accuracy.pdf")

    return model

def findAllFile(base, full=False):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if full:
                yield os.path.join(root, f)
            else:
                yield f

def find_edges(func_key, relations_call, relations_callby, deleted, lv=0, call_type="all", nodes=[], edge_index=[], edge_weight=[]):
    if lv == 0:
        nodes = [func_key]
    func_index = nodes.index(func_key)

    if (lv < 2 and lv > -2) and func_key in relations_call.keys() and call_type in ['all', 'call']:
        weight = {}
        for sub_func in relations_call[func_key]:
            if sub_func in deleted:
                continue
            if not sub_func in weight.keys():
                weight[sub_func] = 1
            else:
                weight[sub_func] += 1
        for sub_func in weight.keys():
            if sub_func not in nodes:
                nodes.append(sub_func)
            sub_func_index = nodes.index(sub_func)
            edge_index.append( [func_index, sub_func_index] )
            edge_weight.append(weight[sub_func])
            nodes, edge_index, edge_weight = find_edges( sub_func, relations_call, relations_callby, deleted, lv+1, "call", nodes, edge_index, edge_weight)

    if (lv < 2 and lv > -2) and func_key in relations_callby.keys() and call_type in ['all', 'callby']:
        weight = {}
        for sub_func in relations_callby[func_key]:
            if sub_func in deleted:
                continue
            if not sub_func in weight.keys():
                weight[sub_func] = 1
            else:
                weight[sub_func] += 0.1
        for sub_func in weight.keys():
            if sub_func not in nodes:
                nodes.append(sub_func)
            sub_func_index = nodes.index(sub_func)
            edge_index.append( [func_index, sub_func_index] )
            edge_weight.append(weight[sub_func])
            nodes, edge_index, edge_weight = find_edges( sub_func, relations_call, relations_callby, deleted, lv-1, "callby", nodes, edge_index, edge_weight)
    return nodes, edge_index, edge_weight


def read_relation_file(relation_file, deleted):
    res = {}

    # read relations
    relations_call = {}
    relations_callby = {}

    json_str = ""
    with open(relation_file) as f:
        json_str = f.read().strip()
    if json_str == "":
        return []
    obj = json.loads(json_str)

    lv0_functions = []
    for k, arr in obj.items():
        for v in arr:
            if v['type'] == 'call':
                if k in relations_call.keys():
                    relations_call[k].append(v['value'])
                else:
                    relations_call[k] = [v['value']]
            elif v['type'] == 'callby':
                if k in relations_callby.keys():
                    relations_callby[k].append(v['value'])
                else:
                    relations_callby[k] = [v['value']]
            elif v['type'] == 'define':
                if v['value'] not in lv0_functions and v['value'] not in deleted:
                    lv0_functions.append(v['value'])
    print("len(lv0_functions):", len(lv0_functions))

    for func in lv0_functions:
        nodes = []
        edge_index = []
        edge_weight = []

        nodes, edge_index, edge_weight = find_edges(func, relations_call, relations_callby, deleted, 0, "all", [], [], [])
        res[func] = {
            "nodes": nodes,
            "edge_index": edge_index,
            "edge_weight": edge_weight
        }
    return res


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
        for f in findAllFile(self.processed_dir, full=True):
            # print("file: ", f)
            if f.find("data_") >= 0 and f.endswith(".pt"):
                pos = f.find("processed") + 10
                # print(f[pos:])
                res.append(f[pos:])

        print("processed_files_num:", len(res))
        return res

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):

        all_funcs_lp_file = self.save_path + "/all_functions_with_trees_lp.csv"

        emb_file_def = self.save_path + "/all_func_embedding_def.csv"
        emb_file_ref = self.save_path + "/all_func_embedding_ref.csv"
        emb_file_pdt = self.save_path + "/all_func_embedding_pdt.csv"

        emb_file_lp_combine = self.save_path + "/all_func_embedding_lp.pkl.combine"
        emb_file_lp_greedy = self.save_path + "/all_func_embedding_lp.pkl.greedy"

        emb_file_ns = self.save_path + "/all_func_embedding_ns.pkl"

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
        func2index = {}
        for index, row in df.iterrows():
            func_key = row['func_key']
            func2index[func_key] = index
            if not (func_key in emb_lp_combine.keys() and func_key in emb_ns.keys()):
                deleted.append( func_key )
        logger.info("=== no lp or ns functions: %d" % len(deleted))

        ii = 0
        with open(TASKS_FILE, 'r') as taskFile:
            taskDesc = json.load(taskFile)
            for repoName in ['nbd','libexpat','jabberd2','redis','ovs','picocom','libvips','libetpan','libreport','neomutt','libksba','miniupnp','imageworsener','mongo-c-driver','mapserver','nettle','rpm','flatpak','torque','yodl','netdata','FreeRDP','rsyslog','libsndfile','pgbouncer','pngquant','lxc']:
                project_path = FUNCTIONS_PATH + "/" + repoName

                for cve_id, non_vul_commits in taskDesc[repoName]['vuln'].items():
                    print(cve_id)
                    relation_file = "%s/%s-relation.json" % (project_path, cve_id)
                    funcs = read_relation_file(relation_file, deleted)
                    for func_key, attr in funcs.items():
                        nodes, edge_index, edge_weight = attr['nodes'], attr['edge_index'], attr['edge_weight']
                        if len(edge_index) == 0:
                            continue

                        edge_weight = torch.tensor(edge_weight, dtype=torch.float)
                        edge_index = torch.tensor(edge_index, dtype=torch.long)
                        if edge_index.shape[0] > 0:
                            edge_index = edge_index - edge_index.min()

                        commit = func_key[:40]
                        y = 0 if commit in non_vul_commits else 1
                        nodes_num = len(nodes)
                        edges_num = len(edge_index)

                        x_lp = torch.zeros([len(nodes), LP_PATH_NUM, LP_LENGTH], dtype=torch.long)
                        x_lp_length = torch.zeros([len(nodes), LP_PATH_NUM], dtype=torch.long)


                        x_ns = torch.zeros([len(nodes), NS_LENGTH], dtype=torch.long)
                        x_ns_length = torch.zeros(len(nodes), dtype=torch.long)


                        x_def = torch.zeros([len(nodes), INPUT_DIM], dtype=torch.float)
                        x_ref = torch.zeros([len(nodes), INPUT_DIM], dtype=torch.float)
                        x_pdt = torch.zeros([len(nodes), INPUT_DIM], dtype=torch.float)
                        for i, sub_func_key in enumerate(nodes):
                            x_lp_word_idx = emb_lp_combine[sub_func_key]['word_idx']
                            for j in range(len(x_lp_word_idx)):
                                if j >= LP_PATH_NUM:
                                    break
                                for k in range(LP_LENGTH):
                                    x_lp[i][j][k] = torch.tensor(x_lp_word_idx[j][k], dtype=torch.long)
                                    x_lp_length[i][j] = max(emb_lp_combine[sub_func_key]['path_length'][j], 1)


                            x_ns[i] = torch.tensor(emb_ns[sub_func_key]['word_idx'], dtype=torch.long)
                            x_ns_length[i] = max(emb_ns[sub_func_key]['ns_length'], 1)

                            func_index = func2index[sub_func_key]
                            x_def[i] = torch.tensor(df_emb_def[func_index], dtype=torch.float)
                            x_ref[i] = torch.tensor(df_emb_ref[func_index], dtype=torch.float)
                            x_pdt[i] = torch.tensor(df_emb_pdt[func_index], dtype=torch.float)

                        data = Data(num_nodes=x_ns.shape[0],
                                    edge_index=edge_index.t().contiguous(),
                                    edge_weight=edge_weight,
                                    y=y,
                                    x_def=x_def,
                                    x_ref=x_ref,
                                    x_pdt=x_pdt,
                                    x_lp=x_lp,
                                    x_lp_length = x_lp_length,
                                    x_ns=x_ns,
                                    x_ns_length = x_ns_length
                                    )

                        # if self.pre_filter is not None and not self.pre_filter(data):
                        #    continue

                        # if self.pre_transform is not None:
                        #    data = self.pre_transform(data)
                        save_path = os.path.join(self.processed_dir, str(ii // 1000))
                        Path(save_path).mkdir(parents=True, exist_ok=True)

                        to_file = os.path.join(save_path, 'data_{}.pt'.format(ii))
                        torch.save(data, to_file)
                        print("saved to: %s, nodes_num: %d, edges_num: %d" % (to_file, nodes_num, edges_num))
                        logger.info("saved to: %s, nodes_num: %d, edges_num: %d" % (to_file, nodes_num, edges_num))
                        ii += 1




    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        save_path = os.path.join(self.processed_dir, str(idx // 1000))
        data = torch.load(os.path.join(save_path, 'data_{}.pt'.format(idx)))
        return data

if __name__ == '__main__':
    writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

    task = 'graph'

    dataset = MyLargeDataset(FUNCTIONS_PATH)
    model = train(dataset, task, writer)
    logger.info("training GCN done.")