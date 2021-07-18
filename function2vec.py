"""


@Time    : 6/21/21
@Author  : Wenbo
"""


import pandas as pd
import argparse
import json
import os
import tempfile
import subprocess
import time
import logging
import hashlib
import preprocess_code
import longpath

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = BASE_DIR + "/data/function2vec"


preprocess_output_location = BASE_DIR + "/data/preprocess"
joern_path = "/opt/joern/"

# make dirs
folders = ['data/', 'logs/', 'projects/', 'data/function2vec', "data/preprocess", 'data/function2vec/models']
for f in folders:
    if not os.path.exists(f):
        os.makedirs(f)

# log file
now_time = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(filename)s line: %(lineno)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=BASE_DIR + '/logs/' + now_time + '.log')
logger = logging.getLogger(__name__)

# cve_jsons_path = BASE_DIR + "/data/jsons"

func_id = 0

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname

def find_functions(functions, cve_id, commit_id, vul, lv=0):
    global func_id
    res = []

    for func in functions:
        if func['code'] == "":
            continue
        item = {
            'func_id': func_id,
            'cve_id': cve_id,
            'commit_id': commit_id,
            'func_name': func['func_name'],
            'file_name': func['file_name'],
            'file_loc': func['file_loc'],
            'level': lv,
            'vul': vul,
            'code': func['code'],
            'is_center': 1 if lv==0 else 0,
            'callees': [],
            'callers': [],
        }
        func_id += 1
        res.append(item)
        if len(func['callees']) > 0:
            sub_funcs = find_functions(func['callees'], cve_id, commit_id, vul, lv+1)
            for sub_f in sub_funcs:
                res.append(sub_f)
                item['callees'].append(sub_f['func_id'])

        if len(func['callers']) > 0:
            sub_funcs = find_functions(func['callers'], cve_id, commit_id, vul, lv+1)
            for sub_f in sub_funcs:
                res.append(sub_f)
                item['callers'].append(sub_f['func_id'])
    return res

def get_commits_prev():
    commits_prev = {}
    log_file = "commits_parrent.log"
    with open(log_file, "r") as f:
        for line in f.readlines():
            l = line.strip()
            if l == "":
                continue

            if l.find("===") > -1:
                l = l.replace("===", "").replace(" ", "")
                cur, prev = l.split(",")
                # print(cur, prev)
                if cur not in commits_prev.keys():
                    commits_prev[cur] = prev
    return commits_prev

# 这里统计的规则是：只要有 function body，无论有无 caller、callee 都行。
def read_cve_jsons(folder_path):
    functions = []
    commits_prev = get_commits_prev()
    logger.info("len(commits_prev): %d" % len(commits_prev))
    found_prev = 0
    for filename in findAllFile(folder_path):
        if filename.endswith(".json"):
            print(filename)

            json_str = ""
            with open(filename) as f:
                json_str = f.read().strip()
            if json_str=="":
                continue
            obj = json.loads(json_str)
            cve_id = obj['cve_id']
            for commit in obj['functions']:
                commit_id = commit['commit_id']
                if commit_id in commits_prev.keys():
                    c_prev = commits_prev[commit_id]
                    found_prev += 1
                else:
                    c_prev = commit_id + "^1"
                funcs_before = find_functions(commit['functions_before'], cve_id, c_prev, 1)
                funcs_after = find_functions(commit['functions_after'], cve_id, commit_id, 0)
                functions = functions + funcs_before + funcs_after
            # break
    logger.info("found_prev: %d" % found_prev)
    return functions

def generate_prolog(code):
    if code.strip() == "":
        return ""
    tmp_dir = tempfile.TemporaryDirectory()
    md5_v = hashlib.md5(code.encode()).hexdigest()
    short_filename = "func_" + md5_v + ".cpp"
    with open(tmp_dir.name + "/" + short_filename, 'w') as f:
        f.write(code)
    # print(short_filename)
    logger.info(short_filename)
    subprocess.check_call(["/opt/joern/joern-parse", tmp_dir.name, "--out", tmp_dir.name + "/cpg.bin.zip"])

    tree = subprocess.check_output(
        "cd /opt/joern && ./joern --script joern_cfg_to_dot.sc --params cpgFile=" + tmp_dir.name + "/cpg.bin.zip",
        shell=True,
        universal_newlines=True,
    )
    pos = tree.find("digraph g {")
    print(pos)
    if pos > 0:
        tree = tree[pos:]
    tmp_dir.cleanup()
    return tree


# 遍历所有已经生成 trees 的 function，返回 func_id 列表。
def get_existed_trees_func_id(filename):
    res = []
    if os.path.exists(filename):
        with open(filename, "r") as f:
            for line in f.readlines():
                l = line.strip()
                if l=="":
                    continue
                item = json.loads(l)
                if item['func_id'] not in res:
                    res.append(item['func_id'])
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test for argparse')
    parser.add_argument('--cve_jsons_path', help='cve_jsons_path', type=str, default='ffmpeg_cve_jsons')
    parser.add_argument('--all_functions_file', help='all_functions_file', type=str, default= SAVE_PATH + '/all_functions.csv')
    parser.add_argument('--all_func_trees_file', help='all_func_trees_file', type=str, default= SAVE_PATH + '/all_functions_with_trees.csv')
    parser.add_argument('--all_func_trees_json_file', help='all_func_trees_json_file', type=str, default= SAVE_PATH + '/all_functions_with_trees.json')

    parser.add_argument('--embedding_type', help='embedding_type', type=str, default= 'ref')
    parser.add_argument('--all_func_embedding_file', help='all_func_embedding_file', type=str, default= SAVE_PATH + '/all_func_embedding_ref.csv')

    args = parser.parse_args()
    logger.info("function2vec parameters %s", args)

    all_func_trees_file = args.all_func_trees_file
    all_func_trees_json_file = args.all_func_trees_json_file

    if not os.path.exists(all_func_trees_file):
        # read cve jsons data
        if not os.path.exists(args.all_functions_file):
            data = read_cve_jsons(args.cve_jsons_path)
            df_functions = pd.DataFrame(data)
            df_functions.to_csv(args.all_functions_file, sep=',', index=None)
            print("saved to: ", args.all_functions_file)
            logger.info("saved to: %s" % args.all_functions_file)
        else:
            df_functions = pd.read_csv(args.all_functions_file)

        logger.info("len(df_functions): %d" % len(df_functions))

        # get trees from code using Joern
        existed = get_existed_trees_func_id(all_func_trees_json_file)
        cc = 0

        df_functions['trees'] = ''
        d_index = list(df_functions.columns).index('trees')
        print("d_index:", d_index)

        for index, row in df_functions.iterrows():
            if row['func_id'] in existed:
                continue
            df_functions.iloc[index, d_index] = generate_prolog(row['code'])
            # save trees in a json file
            with open(all_func_trees_json_file, "a") as fw:
                fw.write(row.to_json() + "\n")
                logger.info("saved to: %s" % all_func_trees_json_file )

            if cc % 1000 == 0:
                cmd = "rm /tmp/joern-default*"
                p = os.popen(cmd)
                x = p.read()
            cc += 1
        # save trees in a csv file
        # df_functions = pd.DataFrame(data)
        df_functions.to_csv(all_func_trees_file, sep=',', index=None)
        print("saved to:", all_func_trees_file)
        logger.info("saved to: %s" % all_func_trees_file)

    # preprocess code
    # REF
    embedding_type = args.embedding_type
    if embedding_type == 'ref':
        tmp_directory = tempfile.TemporaryDirectory()
        input_file = args.all_func_trees_file
        graph2vec_input_dir = preprocess_code.preprocess_all_joern_for_graph2vec(input_file, tmp_directory.name, "REF",
                                                                                 "CALL", num_partitions=10)
        output_file = args.all_func_embedding_file
        preprocess_code.run_graph2vec(graph2vec_input_dir, output_file, num_graph2vec_workers=2, num_epoch=10)

    # DEF
    elif embedding_type == 'def':
        tmp_directory = tempfile.TemporaryDirectory()
        input_file = args.all_func_trees_file
        graph2vec_input_dir = preprocess_code.preprocess_all_joern_for_graph2vec(input_file, tmp_directory.name, "REACHING_DEF",
                                                                                 "EVAL_TYPE", num_partitions=10)
        output_file = args.all_func_embedding_file
        preprocess_code.run_graph2vec(graph2vec_input_dir, output_file, num_graph2vec_workers=2, num_epoch=10)

    # PDT
    elif embedding_type == 'pdt':
        tmp_directory = tempfile.TemporaryDirectory()
        input_file = args.all_func_trees_file
        graph2vec_input_dir = preprocess_code.preprocess_all_joern_for_graph2vec(input_file, tmp_directory.name,
                                                                                 "CFG","REF", num_partitions=10, PDT=True)
        output_file = args.all_func_embedding_file
        preprocess_code.run_graph2vec(graph2vec_input_dir, output_file, num_graph2vec_workers=2, num_epoch=10)

    # LP
    elif embedding_type == 'lp':
        tmp_directory = tempfile.TemporaryDirectory()
        input_file = args.all_func_trees_file
        all_functions_with_lp_file = SAVE_PATH + '/all_functions_with_trees_lp.csv'
        if not os.path.exists(all_functions_with_lp_file):
            longpath.preprocess_longpath(input_file, all_functions_with_lp_file)

        output_file = args.all_func_embedding_file
        w2v_lp_model_file_combine = SAVE_PATH + "/models/w2v_lp_combine.bin"
        w2v_lp_model_file_greedy = SAVE_PATH + "/models/w2v_lp_greedy.bin"
        longpath.run_longpath(all_functions_with_lp_file, output_file, w2v_lp_model_file_combine, w2v_lp_model_file_greedy)

    # NS
    elif embedding_type == 'ns':
        input_file = args.all_func_trees_file
        output_file = args.all_func_embedding_file
        w2v_ns_model = SAVE_PATH + "/models/w2v_ns.bin"
        longpath.run_ns(input_file, output_file, w2v_ns_model)




    print("done")
    logger.info("done")