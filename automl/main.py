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


from model import GNNStack, GNNStack2, GNNStack3
from data_loader import MyLargeDataset

# import nni
# from nni.utils import merge_parameter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

logger = None


# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Additional Info when using cuda
if device.type == 'cuda':
    print('#', torch.cuda.get_device_name(0))
    # print('# Memory Usage:')
    # print('# Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    # print('# Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


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



def test(loader, model, is_validation=False):
    model.eval()

    correct = 0
    y_true = []
    y_pred = []
    for data in loader:
        with torch.no_grad():
            pred = model(data)
            # pred = pred.argmax(dim=1)
            label = data.y

        _, pred = torch.max(pred.data, 1)

        y_true.append(label.cpu())
        y_pred.append(pred.cpu())
        correct += pred.eq(label).sum().item()

    total = len(loader.dataset)

    # return correct / total
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    res = {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }
    # if res['f1'] < 0.01:
    #     logger.info("y_true: {}".format(y_true))
    #     logger.info("y_pred: {}".format(y_pred))
    return res

def print_result(phase, score, epoch = -1):
    if phase in ['train', 'vali']:
        myprint("Epoch {}, {}:\tAcc\tR\tP\tF1\tTN\tFP\tFN\tTP".format(epoch, phase), 1)
        myprint("Epoch {}, {}:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}\t{}\t{}\t{}".format(epoch, phase,
            score['accuracy'], score['recall'], score['precision'], score['f1'],
            score['tn'], score['fp'], score['fn'], score['tp']), 1)
    else:
        myprint("{}:\tAcc\tR\tP\tF1\tTN\tFP\tFN\tTP".format(phase), 1)
        myprint("{}:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}\t{}\t{}\t{}".format(phase,
            score['accuracy'], score['recall'], score['precision'], score['f1'],
            score['tn'], score['fp'], score['fn'], score['tp']), 1)


def train(params, dataset, writer, plot=False, print_grad=False):

    data_size = len(dataset)
    print("data_size:", data_size)
    logger.info("data_size: {}".format(data_size) )


    loader = DataLoader(dataset[:int(data_size * 0.8)], batch_size=params['batch_size'], shuffle=True)
    # loader = DataLoader(dataset[:800], batch_size=params['batch_size'], shuffle=True)

    vali_loader = DataLoader(dataset[int(data_size * 0.8): int(data_size * 0.9)], batch_size=params['batch_size'], shuffle=True)
    test_loader = DataLoader(dataset[int(data_size * 0.9):], batch_size=params['batch_size'], shuffle=True)


    # build model
    if params['model_type'] == 'GNNStack2':
        logger.info("loading model: GNNStack2")
        model = GNNStack2(params)
    elif params['model_type'] == 'GNNStack3':
        logger.info("loading model: GNNStack3")
        model = GNNStack3(params)
    else:
        logger.info("loading model: GNNStack")
        model = GNNStack(params)

    # gpu_tracker = MemTracker()  # define a GPU tracker
    # gpu_tracker.track()
    # print(model)
    # exit()
    model.to(device)

    # gpu_tracker.track()

    # print("len(model.convs):", len(model.convs))

    opt = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=5e-4)
    # opt = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    min_valid_loss = np.inf
    best_f1_score = -1
    best_model_path = ""

    train_accuracies, vali_accuracies = list(), list()

    # train
    for epoch in range(params['epoch']):
        logger.info("=== now epoch: %d" % epoch)
        total_loss = 0
        model.train()
        for batch in loader:
            # print(batch.train_mask, '----')
            opt.zero_grad() # 清空梯度
            pred = model(batch)

            # pred = pred.argmax(dim=1, keepdim=True)
            # logger.info("pred: {}".format(pred))
            label = batch.y.to(device)

            # print("pred.shape : ",pred.shape)
            # print("label.shape: ",label.shape)

            loss = model.loss(pred, label)

            # gpu_tracker.track()

            loss.backward() # 反向计算梯度，累加到之前梯度上
            opt.step() # 更新参数
            total_loss += loss.item() * batch.num_graphs

            # gpu_tracker.track()

            # delete caches
            del pred, loss
            # torch.cuda.empty_cache()

            # gpu_tracker.track()

        total_loss /= len(loader.dataset)
        writer.add_scalar("loss", total_loss, epoch)

        # validate
        train_score = test(loader, model)
        vali_score = test(vali_loader, model)

        train_accuracies.append(train_score['accuracy'])
        vali_accuracies.append(vali_score['accuracy'])

        if print_grad:
            for i, para in enumerate(model.parameters()):
                print(f'{i + 1}th parameter tensor:', para.shape)
                print(para)
                print("grad:")
                print(para.grad)

        logger.info("Epoch: {}, loss: {:.6f}".format(epoch, total_loss))
        if total_loss < min_valid_loss:
            # Saving State Dict
            # torch.save(model.state_dict(), MODEL_SAVE_PATH + "/gcn_model_epoch%d.pth" % epoch)
            # best_model_path = MODEL_SAVE_PATH + "/gcn_model_epoch%d.pth" % epoch

            print("Training Loss Decreased: {:.6f} --> {:.6f}.".format(min_valid_loss, total_loss))
            logger.info("Training Loss Decreased: {:.6f} --> {:.6f}.".format(min_valid_loss, total_loss))
            min_valid_loss = total_loss

        if vali_score['f1'] > best_f1_score:
            # Saving State Dict
            torch.save(model.state_dict(), params['model_save_path'] + "/gcn_model_epoch%d.pth" % epoch)
            best_model_path = params['model_save_path'] + "/gcn_model_epoch%d.pth" % epoch

            print("New best F1: {:.4f} --> {:.4f}. Saved model: {}".format(best_f1_score, vali_score['f1'], best_model_path))
            logger.info("New best F1: {:.4f} --> {:.4f}. Saved model: {}".format(best_f1_score, vali_score['f1'], best_model_path))
            best_f1_score = vali_score['f1']

        print_result("train", train_score, epoch)
        print_result("vali", vali_score, epoch)
        # exit()

        # report intermediate result
        # nni.report_intermediate_result(vali_score['accuracy'])

        # logger.info("w_pdt: {}, w_ref: {}, w_def: {}, w_lp: {}, w_ns: {}".format(model.w_pdt, model.w_ref, model.w_def, model.w_lp, model.w_ns))

        writer.add_scalar("test_accuracy", vali_score['accuracy'], epoch)

    # report final result
    # nni.report_final_result(vali_score['accuracy'])

    # Load
    # model = Net()
    if os.path.exists(best_model_path):
        myprint("loading the best model: %s" % best_model_path, 1)
        model.load_state_dict(torch.load(best_model_path))
        test_score = test(test_loader, model)
        print_result("test", test_score)



    if plot:
        plt.plot(train_accuracies, label="Train accuracy")
        plt.plot(vali_accuracies, label="Validation accuracy")
        plt.xlabel("# Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc='upper right')
        # plt.show()
        plt.savefig(params['model_save_path'] + "/gcn_model_accuracy.pdf")

    return model



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
                if weight[sub_func] < 1.5:
                    weight[sub_func] += 0.1
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
                if weight[sub_func] < 1.5:
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
    if not os.path.exists(relation_file):
        return res

    with open(relation_file) as f:
        json_str = f.read().strip()
    if json_str == "":
        return []
    obj = json.loads(json_str)

    lv0_functions = []
    for k, arr in obj.items():
        for v in arr:
            if v['type'] == 'call':
                if k in deleted or v['value'] in deleted:
                    continue
                if k in relations_call.keys():
                    relations_call[k].append(v['value'])
                else:
                    relations_call[k] = [v['value']]
            elif v['type'] == 'callby':
                if k in deleted or v['value'] in deleted:
                    continue
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


def read_jh_relation_file(entities_file, relation_file, deleted):
    res = {}

    # read relations
    relations_call = {}
    relations_callby = {}

    json_str = ""
    if not os.path.exists(relation_file):
        return res

    lv0_functions = []
    with open(entities_file) as f:
        for line in f.readlines():
            l = line.strip()
            if l=="":
                continue
            obj = json.loads(l)
            if obj['level'] == 0 and obj['vul'] == 1:
                lv0_functions.append(obj['func_key'])

    with open(relation_file) as f:
        for line in f.readlines():
            l = line.strip()
            if l == "":
                continue
            obj = json.loads(l)
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

    print("len(lv0_functions):", len(lv0_functions))
    logger.info("len(lv0_functions): {}".format(len(lv0_functions)))

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



def get_params():
    # args
    parser = argparse.ArgumentParser(description='Test for argparse')
    parser.add_argument('--tasks_file', help='tasks_file', type=str, default='/data/function2vec3/tasks.json')
    parser.add_argument('--functions_path', help='functions_path', type=str,
                        default=BASE_DIR + "/data/function2vec3/functions_jy")
    parser.add_argument('--embedding_path', help='embedding_path', type=str, default='/data/function2vec4')
    parser.add_argument('--model_save_path', help='model_save_path', type=str, default='/data/automl_models')
    parser.add_argument('--best_model_path', help='best_model_path', type=str, default="")

    parser.add_argument('--model_type', help='model_type', type=str, default="GNNStack")
    parser.add_argument('--input_dim', help='input_dim', type=int, default=128)
    parser.add_argument('--output_dim', help='output_dim', type=int, default=2)
    parser.add_argument('--hidden_dim', help='hidden_dim', type=int, default=64)
    parser.add_argument('--batch_size', help='hidden_dim', type=int, default=16)
    parser.add_argument('--learning_rate', help='learning_rate', type=float, default=0.001)
    parser.add_argument('--epoch', help='epoch', type=int, default=100)

    parser.add_argument('--lp_path_num', help='lp_path_num', type=int, default=20)
    parser.add_argument('--lp_length', help='lp_length', type=int, default=60)
    parser.add_argument('--lp_dim', help='lp_dim', type=int, default=128)
    parser.add_argument('--lp_w2v_path', help='lp_w2v_path', type=str,
                        default=BASE_DIR + "/data/function2vec2/models/w2v_lp_combine.bin")

    parser.add_argument('--ns_length', help='ns_length', type=int, default=2000)
    parser.add_argument('--ns_dim', help='ns_dim', type=int, default=128)
    parser.add_argument('--ns_w2v_path', help='ns_w2v_path', type=str,
                        default=BASE_DIR + "/data/function2vec2/models/w2v_ns.bin")

    parser.add_argument('--log_msg', help='log_msg', type=str, default="Soft weights + CE-loss")


    args, _ = parser.parse_known_args()
    # print(args)
    return args

if __name__ == '__main__':
    try:

        params = get_params()

        # get parameters form tuner
        # tuner_params = nni.get_next_parameter()
        # logger.debug(tuner_params)
        #params = vars(merge_parameter(params, tuner_params))

        params = vars(params)
        # logger.info("params: {}".format(params))

        # train
        # /xye_data_nobackup/wenbo/dlvp/data/function2vec4/automl_dataset_1v1/

        # log file
        now_time = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
        log_file = "{}/../logs/{}_{}.log".format(BASE_DIR, now_time, params['model_type'])

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(filename)s line: %(lineno)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            filename=log_file)
        logger = logging.getLogger(__name__)

        # Device
        print('# Using device:', device)
        logger.info("=== using device: %s" % str(device))


        dataset_savepath = params['embedding_path'] + "/automl_dataset_1v1"
        Path(dataset_savepath).mkdir(parents=True, exist_ok=True)
        Path(params['model_save_path']).mkdir(parents=True, exist_ok=True)

        dataset = MyLargeDataset(dataset_savepath, params)

        logger.info("=== Train ===")
        writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        model = train(params, dataset, writer, plot=False, print_grad=False)

    except Exception as exception:
        logger.exception(exception)
        raise

    logger.info("training GCN done.")