"""
给出一段 c/cpp 的 code，生成对应的 longpath 和 natural sequence (NS)。


使用 clang 参考：
https://stackoverflow.com/questions/26000876/how-to-solve-the-loading-error-of-clangs-python-binding

配置环境变量：
export DYLD_LIBRARY_PATH=/usr/local/Cellar/llvm/11.1.0/lib/

配置环境变量(PyCharm)：
https://stackoverflow.com/questions/42708389/how-to-set-environment-variables-in-pycharm

@Time    : 6/25/21
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

try:
   import cPickle as pickle
except:
   import pickle

# # This cell might not be needed for you.
if os.path.exists("ubuntu_only.txt"):
    clang.cindex.Config.set_library_file(
        '/usr/lib/llvm-8/lib/libclang-8.so.1'
    )


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

def generate_long_path(ast_root):
    """
    Given a concretised & numbered clang ast, return long paths
    """
    long_path = []
    edgelist = generate_edgelist(ast_root)
    G = nx.DiGraph()
    G.add_edges_from(edgelist)
    leafnodes = [x for x in G.nodes() if G.out_degree(x)==0 and G.in_degree(x)==1]
    rootnode = [x for x in G.nodes() if G.in_degree(x)==0]
    for node in leafnodes:
        long_path.append(nx.shortest_path(G, source=rootnode[0], target=node))
    return long_path


def code2ns(code):
    res = []
    idx = clang.cindex.Index.create()
    tu = idx.parse('tmp.cpp', args=['-std=c++11'],
                   unsaved_files=[('tmp.cpp', code)], options=0)
    for t in tu.get_tokens(extent=tu.cursor.extent):
        print(t.kind, t.spelling, t.location)
        res.append(t.spelling)
    return res

def code2lp(code):
    if code.strip()=="":
        return json.dumps({
            'nodes':[],
            'edge_list': [],
            'long_path': []
        })

    ast_root = generate_ast_roots(code)

    # get edge
    edge_list = generate_edgelist(ast_root)

    # get nodes
    nodes = generate_features(ast_root)

    # get long path
    long_path = generate_long_path(ast_root)

    return json.dumps({
        'nodes':nodes,
        'edge_list': edge_list,
        'long_path': long_path
    })


def preprocess_longpath(input_file, output_file):
    # get longpath
    all_data = pd.read_csv(input_file)
    all_data = all_data.fillna('')

    all_data["long_path_combine"] = None  # [[[[5, 4, 3, 2, 1, 2, 3, 6, 7], [0, 1, 13, 5]], 1],...,...]
    all_data["long_path_greedy"] = None
    all_data["long_path"] = None
    all_data["nodes"] = None
    all_data["edge_list"] = None
    all_data["flaw_loc"] = None
    all_data["longest_path_token_num"] = None
    all_data["path_num"] = None

    total = len(all_data)

    for index, row in all_data.iterrows():
        if index % 1000 == 0:
            print("preprocess_longpath()... now: %d / %d" % (index, total) )

        # get flaw line ?
        flaw_loc = []

        # flaw_line_code = row["lines_before"] # lines_before 只记录被修改的几行 code
        # if not str(flaw_line_code) == "nan":
        #     flaw_line_code = flaw_line_code.strip().split("\n")
        #     for each_flaw_line in range(len(flaw_line_code)):
        #         flaw_line_code[each_flaw_line] = flaw_line_code[each_flaw_line].strip()
        #     code = row["func_before"].strip().split("\n")
        #     for code_line in range(len(code)):
        #         code[code_line] = code[code_line].strip()
        #
        #     for flaw_line in flaw_line_code:
        #         for line_loc, line in enumerate(code):
        #             if flaw_line == line:
        #                 flaw_loc.append(line_loc)
        # flaw_loc = list(set(flaw_loc))
        # if len(flaw_loc) > 0:
        #     all_data.loc[index, "flaw_loc"] = str(flaw_loc)


        # get ast
        code = row["code"]
        code = code.split('\n')
        line_1 = code[0]
        pos = line_1.find("(")
        if line_1.find("::") > 0:
            if pos > 0:
                scope_opr = line_1.rfind("::", 0, pos)
            else:
                scope_opr = line_1.rfind("::")
            class_ = line_1.rfind(" ", 0, scope_opr)
            line_1 = line_1.replace(line_1[class_:scope_opr + 2], " ")
        code[0] = line_1
        code = "\n".join(code)
        ast_root = generate_ast_roots(code)

        # get edge
        edge_list = generate_edgelist(ast_root)

        # get nodes
        nodes = generate_features(ast_root)

        # get long path
        long_path = generate_long_path(ast_root)
        if not ((len(long_path) == 1 and len(long_path[0]) == 2) or (len(long_path) == 0)):
            all_data.loc[index, "long_path"] = str(long_path)
            all_data.loc[index, "edge_list"] = str(edge_list)
            all_data.loc[index, "nodes"] = str(nodes)

            long_path_greedy = sorted(long_path, key=lambda i: len(i), reverse=True)

            path_len = len(long_path_greedy)
            long_path_greedy_2 = []
            long_path_2 = []
            i = 0
            if path_len % 2 == 0:
                while i < path_len:
                    start1 = long_path_greedy[i][::-1]
                    end1 = long_path_greedy[i + 1][1:]
                    path1 = start1 + end1
                    long_path_greedy_2.append(path1)

                    start2 = long_path[i][::-1]
                    end2 = long_path[i + 1][1:]
                    path2 = start2 + end2
                    long_path_2.append(path2)

                    i += 2
            else:
                while i < path_len:
                    if i == path_len - 1:
                        start1 = long_path_greedy[i][::-1]
                        end1 = long_path_greedy[i][1:]
                        path1 = start1 + end1
                        long_path_greedy_2.append(path1)

                        start2 = long_path[i][::-1]
                        end2 = long_path[i][1:]
                        path2 = start2 + end2
                        long_path_2.append(path2)
                        i += 2
                    else:
                        start1 = long_path_greedy[i][::-1]
                        end1 = long_path_greedy[i + 1][1:]
                        path1 = start1 + end1
                        long_path_greedy_2.append(path1)

                        start2 = long_path[i][::-1]
                        end2 = long_path[i + 1][1:]
                        path2 = start2 + end2
                        long_path_2.append(path2)

                        i += 2

            long_path_greedy_2_with_cover_lines = []
            long_path_2_with_cover_lines = []

            for each_path in long_path_greedy_2:
                cover_lines = []
                for node_id in each_path:
                    cover_lines.append(nodes[node_id][3])
                long_path_greedy_2_with_cover_lines.append([each_path, list(set(cover_lines))])

            for each_path in long_path_2:
                cover_lines = []
                for node_id in each_path:
                    cover_lines.append(nodes[node_id][3])
                long_path_2_with_cover_lines.append([each_path, list(set(cover_lines))])

            long_path_greedy_2_with_label = []
            long_path_2_with_label = []

            for each_path in long_path_greedy_2_with_cover_lines:
                loc_find = False
                for node_id in each_path[0]:
                    if nodes[node_id][3] in flaw_loc:
                        loc_find = True
                if loc_find:
                    long_path_greedy_2_with_label.append([each_path, 1])
                else:
                    long_path_greedy_2_with_label.append([each_path, 0])

            for each_path in long_path_2_with_cover_lines:
                loc_find = False
                for node_id in each_path[0]:
                    if nodes[node_id][3] in flaw_loc:
                        loc_find = True
                if loc_find:
                    long_path_2_with_label.append([each_path, 1])
                else:
                    long_path_2_with_label.append([each_path, 0])

            all_data.loc[index, "long_path_greedy"] = str(long_path_greedy_2_with_label)
            all_data.loc[index, "long_path_combine"] = str(long_path_2_with_label)
            all_data.loc[index, "path_num"] = len(long_path_2_with_label)
            longest_path_token_num = 0
            for path in long_path_2_with_label:
                if len(path[0][0]) > longest_path_token_num:
                    longest_path_token_num = len(path[0][0])
            all_data.loc[index, "longest_path_token_num"] = longest_path_token_num

    all_data.to_csv(output_file, sep=',', index=None)
    print("saved to: %s" % output_file)

def tokens2vec(tokens, w2v):
    length = len(tokens)
    matrix = np.zeros((length, 128)) #word embedding size is 100
    for i, ww in enumerate(tokens):
        if i >= length:
            break
        if ww in w2v.wv.vocab:
            matrix[i] = np.array(w2v.wv[ww])

    #l2 normalize
    try:
        norm = np.linalg.norm(matrix, axis=1).reshape(length, 1)
        matrix = np.divide(matrix, norm, out=np.zeros_like(matrix), where=norm!=0)
        #matrix = matrix / np.linalg.norm(matrix, axis=1).reshape(len(doc), 1)
    except RuntimeWarning:
        print(tokens)

    #matrix = np.array(preprocessing.normalize(matrix, norm='l2'))
    return np.array(np.mean(matrix, axis=0))

def run_longpath(input_file, output_file, w2v_lp_model_file_combine, w2v_lp_model_file_greedy):
    # train word2vec
    all_data = pd.read_csv(input_file)
    all_data = all_data.fillna('')

    all_data["long_path_greedy_context"] = None
    all_data["long_path_combine_context"] = None

    corpus_long_path_combine_context = []
    corpus_long_path_greedy_context = []
    for index, row in all_data.iterrows():
        if row["long_path_combine"]=='' or row["long_path_greedy"]=='' or row["nodes"]=='':
            continue

        long_path_combine = eval(row["long_path_combine"])
        long_path_greedy = eval(row["long_path_greedy"])
        nodes = eval(row["nodes"])

        long_path_combine_all_context = []
        long_path_combine_each_context = []
        for path in long_path_combine:
            each_context = []
            len_path = len(path[0][0])
            i = 0
            for node_id in path[0][0]:
                if i == 0:
                    long_path_combine_all_context.extend(nodes[node_id][2].split())
                    each_context.append(nodes[node_id][2].split())
                elif i == len(path[0][0]) - 1:
                    long_path_combine_all_context.extend(nodes[node_id][2].split())
                    each_context.append(nodes[node_id][2].split())
                else:
                    long_path_combine_all_context.append(nodes[node_id][1])
                    each_context.append(nodes[node_id][1])
                i += 1
            long_path_combine_each_context.append([[each_context, path[0][1]], path[1]])

        long_path_greedy_all_context = []
        long_path_greedy_each_context = []
        for path in long_path_greedy:
            each_context = []
            len_path = len(path[0][0])
            i = 0
            for node_id in path[0][0]:
                if i == 0:
                    long_path_greedy_all_context.extend(nodes[node_id][2].split())
                    each_context.append(nodes[node_id][2].split())
                elif i == len(path[0][0]) - 1:
                    long_path_greedy_all_context.extend(nodes[node_id][2].split())
                    each_context.append(nodes[node_id][2].split())
                else:
                    long_path_greedy_all_context.append(nodes[node_id][1])
                    each_context.append(nodes[node_id][1])
                i += 1
            long_path_greedy_each_context.append([[each_context, path[0][1]], path[1]])

        all_data.loc[index, "long_path_greedy_context"] = str(long_path_greedy_each_context)
        all_data.loc[index, "long_path_combine_context"] = str(long_path_combine_each_context)

        corpus_long_path_combine_context.append(long_path_combine_all_context)
        corpus_long_path_greedy_context.append(long_path_greedy_all_context)
        if index % 1000 == 0:
            print(index)

    w2v_long_path_combine_context = Word2Vec(corpus_long_path_combine_context, size=128, workers=16, sg=1, min_count=1)
    w2v_long_path_combine_context.save(w2v_lp_model_file_combine)
    print("saved to: %s" % w2v_lp_model_file_combine)

    w2v_long_path_greedy_context = Word2Vec(corpus_long_path_greedy_context, size=128, workers=16, sg=1, min_count=1)
    w2v_long_path_greedy_context.save(w2v_lp_model_file_greedy)
    print("saved to: %s" % w2v_lp_model_file_greedy)

    # get embeddings
    to_data_combine = {}
    to_data_greedy = {}

    # greedy_path_embedding = []
    # combine_path_embedding = []
    for index, row in all_data.iterrows():
        func_id = row['func_id']

        if row['long_path_greedy_context'] == '' or row["long_path_combine_context"]=='':
            continue

        try:
            long_path_greedy_context = eval(row["long_path_greedy_context"])
            long_path_combine_context = eval(row["long_path_combine_context"])
        except:
            continue

        # get max path length
        max_path_length = -1
        for path in long_path_greedy_context:
            max_path_length = max(max_path_length, len(path[0][0]))
        for path in long_path_combine_context:
            max_path_length = max(max_path_length, len(path[0][0]))
        print("max_path_length: ", max_path_length)
        logging.info("== max_path_length: %d "% max_path_length)


        function_paths_greedy = []
        for path in long_path_greedy_context:
            # path_embedding = [] # matrix
            path_embedding = np.zeros((max_path_length, 128))
            for i, token in enumerate( path[0][0] ):
                if type(token).__name__ == 'list':
                    logging.info("== token is list?: yes")
                    print("== token is list?: yes")
                    sum_avg = tokens2vec(token, w2v_long_path_greedy_context)
                    if not len(token) == 0:
                        path_embedding[i] = sum_avg
                else:
                    logging.info("== token is list?: no")
                    print("== token is list?: no")
                    if token in w2v_long_path_greedy_context.wv.vocab:
                        # path_embedding.append(w2v_long_path_greedy_context.wv.vocab[token])
                        path_embedding[i] = w2v_long_path_greedy_context.wv[token]
            function_paths_greedy.append(path_embedding)
        to_data_greedy[func_id] = function_paths_greedy
        # greedy_path_embedding.append(function_paths_greedy)


        function_paths_combine = []
        for path in long_path_combine_context:
            # path_embedding = []
            path_embedding = np.zeros((max_path_length, 128))
            for i,token in enumerate(path[0][0]):
                if type(token).__name__ == 'list':
                    sum_avg = tokens2vec(token, w2v_long_path_combine_context)
                    if not len(token) == 0:
                        # path_embedding.append(sum_avg)
                        path_embedding[i] = sum_avg
                else:
                    if token in w2v_long_path_combine_context.wv.vocab:
                        # path_embedding.append(w2v_long_path_combine_context.wv.vocab[token])
                        path_embedding[i] = w2v_long_path_greedy_context.wv[token]
            function_paths_combine.append(path_embedding)
        # combine_path_embedding.append(function_paths_combine)
        to_data_combine[func_id] = function_paths_combine
    to_file_combine = output_file + ".combine"
    to_file_greedy = output_file + ".greedy"

    pickle.dump(to_data_combine, open(to_file_combine, "wb"))
    pickle.dump(to_data_greedy, open(to_file_greedy, "wb"))
    print("saved to %s" % to_file_combine)
    print("saved to %s" % to_file_greedy)



if __name__ == '__main__':

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

    # get long path
    long_path = generate_long_path(ast_root)
    print("=== long_path:")
    print(long_path)

    pass