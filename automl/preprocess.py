"""
functions --> embeddings.

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



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = BASE_DIR + "/../logs"
Path(LOG_PATH).mkdir(parents=True, exist_ok=True)

# log file
now_time = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(filename)s line: %(lineno)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=LOG_PATH + '/' + now_time + '_preprocess.log')
logger = logging.getLogger(__name__)

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('# Using device:', device)
logger.info("=== using device: %s" % str(device))

# Additional Info when using cuda
if device.type == 'cuda':
    print('#', torch.cuda.get_device_name(0))
    # print('# Memory Usage:')
    # print('# Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    # print('# Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

# args
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--tasks_file', help='tasks_file', type=str, default=BASE_DIR + '/../cflow/tasks.json')
parser.add_argument('--functions_path', help='functions_path', type=str,
                    default=BASE_DIR + "/../data/function2vec2/functions_jy")
parser.add_argument('--embedding_path', help='embedding_path', type=str, default=BASE_DIR + "/../data/function2vec2")



args = parser.parse_args()

logger.info("gcn parameters %s", args)

EMBEDDING_PATH = args.embedding_path
FUNCTIONS_PATH = args.functions_path
TASKS_FILE = args.tasks_file


def findAllFile(base, full=False):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if full:
                yield os.path.join(root, f)
            else:
                yield f


def find_edges(stats, func_key, relations_call, relations_callby, deleted, lv=0, call_type="all", nodes=[], edge_index=[],
               edge_weight=[]):
    if lv == 0:
        nodes = [func_key]
    func_index = nodes.index(func_key)
    if lv in stats.keys():
        stats[lv] += 1
    else:
        stats[lv] = 1

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
            edge_index.append([func_index, sub_func_index])
            edge_weight.append(weight[sub_func])
            stats, nodes, edge_index, edge_weight = find_edges(stats, sub_func, relations_call, relations_callby, deleted, lv + 1,
                                                        "call", nodes, edge_index, edge_weight)

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
            edge_index.append([func_index, sub_func_index])
            edge_weight.append(weight[sub_func])
            stats, nodes, edge_index, edge_weight = find_edges(stats, sub_func, relations_call, relations_callby, deleted, lv - 1,
                                                        "callby", nodes, edge_index, edge_weight)
    return stats, nodes, edge_index, edge_weight


def read_relation_file(non_vul_commits, repoName, relation_file, deleted):
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
        commit_id = func[:40]
        # print("commit_id:", commit_id)
        nodes = []
        edge_index = []
        edge_weight = []
        stats = {}

        stats, nodes, edge_index, edge_weight = find_edges(stats, func, relations_call, relations_callby, deleted, 0, "all", [], [],
                                                    [])
        res[func] = {
            "repo_name": repoName,
            "nodes": nodes,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "stats": stats,
            "vul": 0 if commit_id in non_vul_commits else 1
        }
    return res


def read_jh_relation_file(finished_repos, entities_file, relation_file, deleted):
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
            if l == "":
                continue
            obj = json.loads(l)
            if obj['level'] == 0 and obj['repo_name'] not in finished_repos:
                lv0_functions.append( (obj['func_key'], obj['repo_name'], obj['vul']) )

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

    for func, repo_name, vul in lv0_functions:
        nodes = []
        edge_index = []
        edge_weight = []
        stats = {}

        stats, nodes, edge_index, edge_weight = find_edges(stats, func, relations_call, relations_callby, deleted, 0, "all", [], [],
                                                    [])
        res[func] = {
            "repo_name": repo_name,
            "nodes": nodes,
            "stats": stats,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "vul": vul
        }
    return res


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

def get_graphs(funcs):
    res = []
    for func_key, attr in funcs.items():
        if len(attr['edge_index']) == 0:
            res.append({
                'func_key': func_key,
                'repo_name': attr['repo_name'],
                'stats': attr['stats'],
                'vul': attr['vul'],
                'unique_funcs': len(attr['nodes']),
                'G': None
            })
        else:
            G = nx.DiGraph()  # directed graph
            nodes = attr['nodes']
            G.add_nodes_from([n for n in nodes])

            edges = []
            for row in attr['edge_index']:
                n1 = nodes[ row[0] ]
                n2 = nodes[ row[1] ]
                edges.append( (n1, n2) )
            G.add_edges_from(edges)
            res.append({
                'func_key': func_key,
                'repo_name': attr['repo_name'],
                'stats': attr['stats'],
                'vul': attr['vul'],
                'unique_funcs': len(nodes),
                'G': G
            })
    return res

def log_stats(graphs, only_vul = False):
    logger.info("all functions: {}".format(len(graphs)))

    to_data = []
    for g in graphs:
        if only_vul and g['vul'] == 0:
            continue

        stats = g['stats']
        item = {
            'Project': g['repo_name'],
            'LV0': stats[0] if 0 in stats.keys() else 0,

            'LV1(Callees)': stats[1] if 1 in stats.keys() else 0,
            'LV1(Callers)': stats[-1] if -1 in stats.keys() else 0,

            'LV2(Callees)': stats[2] if 2 in stats.keys() else 0,
            'LV2(Callers)': stats[-2] if -2 in stats.keys() else 0,

            'Unique_Funcs': g['unique_funcs']
        }

        to_data.append(item)



    to_df = pd.DataFrame(to_data)

    save_path = BASE_DIR + "/data"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    if only_vul:
        to_file = save_path + "/dataset_stats_only_vul.csv"
    else:
        to_file = save_path + "/dataset_stats_all.csv"

    to_df = to_df.groupby('Project').sum()
    to_df = to_df.sort_values(by='LV0', ascending=False)
    to_df.to_csv(to_file, sep=',')
    print("saved to {}".format(to_file))
    logger.info("saved to {}".format(to_file))

    if only_vul:
        all_vul_functions = 0
        lv_0 = 0
        lv_1 = 0
        lv_2 = 0
        lv_n1 = 0
        lv_n2 = 0
        unique = 0
        for index, row in to_df.iterrows():
            all_vul_functions += row['LV0']

            if row['LV1(Callees)'] + row['LV1(Callers)'] > 0:
                lv_0 += row['LV0']
                lv_1 += row['LV1(Callees)']
                lv_n1 += row['LV1(Callers)']

                lv_2 += row['LV2(Callees)']
                lv_n2 += row['LV2(Callers)']

                unique += row['Unique_Funcs']

        logger.info("all vul functions: {}".format(all_vul_functions))
        logger.info("lv_0: {}".format(lv_0))
        logger.info("lv_1: {}".format(lv_1))
        logger.info("lv_n1: {}".format(lv_n1))
        logger.info("lv_2: {}".format(lv_2))
        logger.info("lv_n2: {}".format(lv_n2))
        logger.info("Unique_Funcs: {}".format(unique))


def process():
    all_func_trees_json_file_merged = EMBEDDING_PATH + '/all_functions_with_trees_merged_jh.json'

    # read jy's data
    ii = 0
    graphs = []
    finished_repos = []
    with open(TASKS_FILE, 'r') as taskFile:
        taskDesc = json.load(taskFile)

        for repoName in taskDesc:
            finished_repos.append(repoName)
            project_path = FUNCTIONS_PATH + "/" + repoName

            for cve_id, non_vul_commits in taskDesc[repoName]['vuln'].items():
                print(cve_id)
                relation_file = "%s/%s-relation.json" % (project_path, cve_id)
                if not os.path.exists(relation_file):
                    logger.info("file not existed: %s" % relation_file)
                    continue

                funcs = read_relation_file(non_vul_commits, repoName, relation_file, [])
                graphs = graphs + get_graphs( funcs )
    logger.info("len(jy's data): {}".format(len(graphs)))

    # read jh's data
    JIAHAO_DATA_PATH = "/data/jiahao_data"
    jh_entities_file = JIAHAO_DATA_PATH + "/jh_entities.json"
    jh_relations_file = JIAHAO_DATA_PATH + "/jh_relations.json"
    if os.path.exists(jh_entities_file):
        logger.info("reading jh's data")
        funcs = read_jh_relation_file(finished_repos, jh_entities_file, jh_relations_file, [])
        graphs = graphs + get_graphs(funcs)

    # 统计数据集
    log_stats(graphs, only_vul=True)
    log_stats(graphs)


if __name__ == '__main__':
    process()