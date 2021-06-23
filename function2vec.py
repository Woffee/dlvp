"""


@Time    : 6/21/21
@Author  : Wenbo
"""


import pandas as pd
import argparse
import json
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = BASE_DIR + "/data/function2vec"

# make dirs
folders = ['data/', 'logs/', 'projects/', 'data/function2vec']
for f in folders:
    if not os.path.exists(f):
        os.makedirs(f)

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
        item = {
            'func_id': func_id,
            'cve_id': cve_id,
            'commit_id': commit_id,
            'func_name': func['func_name'],
            'file_name': func['file_name'],
            'file_loc': func['file_loc'],
            'code': func['code'],
            'is_center': 1 if lv==0 else 0,
            'callees': [],
            'callers': [],
        }
        func_id += 1
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
        res.append(item)
    return res

def read_cve_jsons(folder_path):
    functions = []

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
                funcs_before = find_functions(commit['functions_before'], cve_id, commit_id, 1)
                funcs_after = find_functions(commit['functions_after'], cve_id, commit_id, 0)
                functions = functions + funcs_before + funcs_after
            # break
    return functions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test for argparse')
    parser.add_argument('--cve_jsons_path', help='cve_jsons_path', type=str, default='ffmpeg_cve_jsons')
    parser.add_argument('--all_functions_file', help='all_functions_file', type=str, default= SAVE_PATH + '/all_functions.csv')

    args = parser.parse_args()


    # read cve jsons data
    data = read_cve_jsons(args.cve_jsons_path)
    df_functions = pd.DataFrame(data)
    if not os.path.exists(args.all_functions_file):
        df_functions.to_csv(args.all_functions_file, sep=',', index=None)
        print("saved to: ", args.all_functions_file)




