"""
把收集到的所有 vulnerable 的 functions

@Time    : 8/5/21
@Author  : Wenbo
"""

import networkx as nx
import os
import clang.cindex
import json
import pandas as pd
from gensim.models.word2vec import Word2Vec
import numpy as np
import logging
from pathlib import Path
import time
import argparse


try:
   import cPickle as pickle
except:
   import pickle

# # This cell might not be needed for you.
if os.path.exists("/usr/lib/llvm-8/lib/libclang-8.so.1"):
    clang.cindex.Config.set_library_file(
        '/usr/lib/llvm-8/lib/libclang-8.so.1'
    )


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# log file
LOG_PATH = BASE_DIR + "/logs"
Path(LOG_PATH).mkdir(parents=True, exist_ok=True)

now_time = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(filename)s line: %(lineno)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=LOG_PATH + '/' + now_time + '_prepare_data.log')
logger = logging.getLogger(__name__)


# args
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--entities_file', help='entities_file', type=str, default='')
parser.add_argument('--tasks_file', help='tasks_file', type=str, default=BASE_DIR + '/../cflow/tasks.json')
parser.add_argument('--functions_path', help='functions_path', type=str, default=BASE_DIR + "/../data/function2vec2/functions_jy")
parser.add_argument('--save_path', help='save_path', type=str, default=BASE_DIR + '/data')

args = parser.parse_args()
logger.info("treecaps/prepare_data parameters %s", args)

SAVE_PATH = args.save_path
Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)

ENTITIES_FILE = args.entities_file
TASKS_FILE = args.tasks_file
FUNCTIONS_PATH = args.functions_path


# init vars
all_node_types = []


def generate_ast_roots(code):
    """
    Takes in a list of files/datapoints from juliet.csv.zip (as loaded with pandas) matching one particular
    testcase, and preprocesses it ready for the feature matrix.
    """
    index = clang.cindex.Index.create()
    parse_list = [('test.cpp', code)]
    translation_unit = index.parse(
        path='test.cpp',
        unsaved_files=parse_list,
    )
    ast_root = translation_unit.cursor

    concretise_ast(ast_root)
    number_ast_nodes(ast_root)

    return ast_root


def generate_features(ast_root):
    """
    Given a concretised & numbered clang ast, return a dictionary of
    features in the form:
        {
            <node_id>: [<degree>, <type>, <identifier>, <line_num>],
            ...
        }
    """
    features = {}

    def walk_tree_and_set_features(node):
        out_degree = len(node.children)
        in_degree = 1
        degree = out_degree + in_degree

        features[node.identifier] = [degree, str(node.kind), node.displayname, node.location.line]

        for child in node.children:
            walk_tree_and_set_features(child)

    walk_tree_and_set_features(ast_root)

    return features

def concretise_ast(node):
    """
    Everytime you run .get_children() on a clang ast node, it
    gives you new objects. So if you want to modify those objects
    they will lose their changes everytime you walk the tree again.
    To avoid this problem, concretise_ast walks the tree once,
    saving the resulting list from .get_children() into a a concrete
    list inside the .children.
    You can then use .children to consistently walk over tree, and
    it will give you the same objects each time.
    """
    node.children = list(node.get_children())

    for child in node.children:
        counter = concretise_ast(child)


def number_ast_nodes(node, counter=1):
    """
    Given a concretised clang ast, assign each node with a unique
    numerical identifier. This will be accessible via the .identifier
    attribute of each node.
    """
    node.identifier = counter
    counter += 1

    node.children = list(node.get_children())
    for child in node.children:
        counter = number_ast_nodes(child, counter)

    return counter

def generate_edgelist(ast_root):
    """
    Given a concretised & numbered clang ast, return a list of edges
    in the form:
        [
            [<start_node_id>, <end_node_id>],
            ...
        ]
    """
    edges = []

    def walk_tree_and_add_edges(node):
        for child in node.children:
            edges.append([node.identifier, child.identifier])
            walk_tree_and_add_edges(child)

    walk_tree_and_add_edges(ast_root)

    return edges

def generate_ast_tree(ast_root, label):


    def walk_tree_and_add_edges(node):
        node_type = str(node.kind)

        if node.kind not in all_node_types:
            all_node_types.append( node_type )
        node_id = str( all_node_types.index( node_type ) )
        cur_node = {
            # 'node': str(node.identifier),
            'node': node_id,
            'children': []
        }
        for child in node.children:
            cur_node['children'].append(walk_tree_and_add_edges(child))
        return cur_node

    return {
        'tree': walk_tree_and_add_edges(ast_root),
        'label': label
    }

def generate_node_type(ast_root):
    node_types = {}

    def walk_tree_and_set_features(node):
        out_degree = len(node.children)
        in_degree = 1
        degree = out_degree + in_degree

        # node_types[ str(node.identifier) ] = [degree, str(node.kind), node.displayname, node.location.line]
        node_types[ str(node.identifier) ] = str(node.kind)

        for child in node.children:
            walk_tree_and_set_features(child)

    walk_tree_and_set_features(ast_root)
    return node_types


def test():
    code = """
    int who_am_i (void)
    {
      struct passwd *pw;
      char *user = NULL;

      pw = getpwuid (geteuid ());
      if (pw)
        user = pw->pw_name;
      else if ((user = getenv ("USER")) == NULL)
        {
          fprintf (stderr, "I don't know!\n");
          return 1;
        }
      printf ("%s\n", user);

      fac(5);
      return 0;
    }

    int fac(int n) {
        return (n>1) ? n*fac(n-1) : 1;
    }
    """
    ast_root = generate_ast_roots(code)
    print("=== ast:")
    print(ast_root)

    # get edge
    edge_list = generate_edgelist(ast_root)
    print("=== edge_list:")
    print(edge_list)

    # get nodes
    nodes = generate_features(ast_root)
    print("=== nodes:")
    print(nodes)

    # tree for treecaps
    ast_tree = generate_ast_tree(ast_root, '1')
    print("=== ast_tree:")
    print(ast_tree)

    node_types = generate_node_type(ast_root)
    print("=== node_types:")
    print(node_types)



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

def read_functions(repo_name, entities_filename, non_vul_commits):
    """
    读取一个 entities 文件里的所有 functions。

    """
    functions = []
    cve_id = entities_filename.split("/")[-1].replace("-entities.json", "")

    # read entities
    json_str = ""
    with open(entities_filename) as f:
        json_str = f.read().strip()
    if json_str == "":
        return []
    cve_functions = json.loads(json_str)

    for k, v in cve_functions.items():
        vul = 0 if v['commit'] in non_vul_commits else 1

        functions.append({
            'repo_name': repo_name,
            'cve_id': cve_id,
            'commit': v['commit'],
            'func_key': k,
            'name': v['name'],
            'uniquename': v['uniquename'],
            'contents': v['contents'],
            'kind': v['kind'],
            'type': v['type'],
            'vul': vul,
        })

    return functions


def read_lv0_functions(relation_file, deleted):
    res = {}

    # read relations
    relations_call = {}
    relations_callby = {}

    json_str = ""
    if not os.path.exists(relation_file):
        logger.info("file not exists: {}".format(relation_file))
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
                if v['value'] in deleted:
                    logger.info("lv0_function no LP or NS: {}".format(v['value']))
                    continue
                if v['value'] not in lv0_functions:
                    lv0_functions.append(v['value'])

    return lv0_functions


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



if __name__ == '__main__':
    # test()

    deleted = []
    funcs_cnt = 0
    ii = 0
    with open(TASKS_FILE, 'r') as taskFile:
        taskDesc = json.load(taskFile)

        top10 = ['FFmpeg','qemu','openssl','tcpdump','php-src','kerberos_5','jasper','librsvg','freetype','radare2']
        for repoName in taskDesc:
            if repoName not in top10:
                continue

            i_repo = top10.index(repoName)
            print("now: {}".format(repoName))
            logger.info("now: {}".format(repoName))

            # sub_save_path = SAVE_PATH + "/" + str(i_repo)
            # Path(sub_save_path).mkdir(parents=True, exist_ok=True)
            to_file_trees = "{}/treecaps_trees_{}.pkl".format(SAVE_PATH, i_repo)
            if os.path.exists(to_file_trees):
                continue
            trees = []

            project_path = FUNCTIONS_PATH + "/" + repoName
            for cve_id, non_vul_commits in taskDesc[repoName]['vuln'].items():
                print(cve_id)

                entities_file = "%s/%s-entities.json" % (project_path, cve_id)
                relation_file = "%s/%s-relation.json" % (project_path, cve_id)

                if not os.path.exists(relation_file):
                    logger.info("file not existed: %s" % relation_file)
                    continue

                cve_functions = read_functions(repoName, entities_file, non_vul_commits)
                lv0_functions = read_lv0_functions(relation_file, deleted)

                for func in cve_functions:
                    if func['func_key'] not in lv0_functions:
                        continue
                    logger.info("func_key: {}".format(func['func_key']))

                    ast_root = generate_ast_roots( func['contents'] )
                    trees.append( generate_ast_tree(ast_root, str(i_repo)) )


            funcs_cnt += len(trees)
            with open(to_file_trees, 'wb') as file_handler:
                pickle.dump(trees, file_handler, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("saved to {}, len(trees): {}".format(to_file_trees, len(trees)))

    node_type_lookup = {}
    to_file_node_type = SAVE_PATH + "/node_type_lookup.pkl"
    for i, v in enumerate(all_node_types):
        node_type_lookup[ str(i) ] = v

    with open(to_file_node_type, 'wb') as file_handler:
        pickle.dump(node_type_lookup, file_handler, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("saved to {}".format(to_file_node_type))

    # with open(ENTITIES_FILE, "r") as f:
    #     for line in f:
    #         if line.strip() == "":
    #             continue
    #
    #         obj = json.loads(line)