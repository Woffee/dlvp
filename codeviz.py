
"""

May 17, 2021:
 - 对于每个 vulnerability，找出每个 function，在每个 commit 前后的对比。记录 caller、callee。如果有重复出现的 function，取最早的 commit 修改之前的版本，和最后的 commit 修改之后的版本，做对比。
 - 忽略 test 下的文件。



git --no-pager show e43a0a232dbf6d3c161823c2e07c52e76227a1bc

cflow --main who_am_i whoami.c



列出 每个 function 的 caller callee
When ‘--all’ is used twice, graphs for all global functions (whether top-level or not) will be displayed.

e.g.

    cflow test_cflow.c --depth=3 --all --all

will output:

```
main() <int main () at test_cflow.c:27>:
    funcA() <int funcA () at test_cflow.c:21>:
        funcB() <int funcB () at test_cflow.c:15>:
funcA() <int funcA () at test_cflow.c:21>:
    funcB() <int funcB () at test_cflow.c:15>:
        funcC() <int funcC () at test_cflow.c:10>:
funcB() <int funcB () at test_cflow.c:15>:
    funcC() <int funcC () at test_cflow.c:10>:
        funcD() <int funcD () at test_cflow.c:5>
funcC() <int funcC () at test_cflow.c:10>:
    funcD() <int funcD () at test_cflow.c:5>
funcD() <int funcD () at test_cflow.c:5>
```



列出 所有 commit log，一行一个 commit id
git --no-pager log --pretty=oneline

查看一个 tag 对应的 commit id
git rev-list -n 1 $TAG


查看 两个 commits 之间的不同：
https://stackoverflow.com/questions/3368590/show-diff-between-commits
git diff oldCommit newCommit
git --no-pager diff ed188f6dcdf0935c939ed813cf8745d50742014b 02f909dc24b1f05cfbba75077c7707b905e63cd2



# 【1】找出 修改的 filename 和 func name

# 【2】列举所有 c 文件 --> 列举所有的文件路径

# 【3】cflow path/*.c

# 【4】找出【1】中的 caller 和 callee

# 【5】然后，最这些 caller 和 callee 提取代码，即为 code_after, caller_after, callee_after


# git reset 到本次修改前

# 重复 【2】- 【5】，即为 code_before, caller_before, callee_before




@Time    : 5/8/21
@Author  : Wenbo
"""


import os
import re
import time
import logging
import argparse
import pandas as pd
import json


data_path = "mydata"
data_type = "ffmpeg"

ffmpeg_commits_file = data_path + "/ffmpeg_commits_in_order_with_fixed_tag.csv"

show_log_savepath = data_path + "/ffmpeg"
if not os.path.exists(show_log_savepath):
    os.makedirs(show_log_savepath)

json_savepath = data_path + "/ffmpeg_jsons"
if not os.path.exists(json_savepath):
    os.makedirs(json_savepath)

to_df_file = data_path + "/ffmpeg_caller_callee.csv"


class Function:
    def __init__(self, func_name, file_name="", file_loc=0, code=""):
        self.func_name = func_name
        self.file_name = file_name
        self.file_loc = file_loc
        self.code = code
        self.callees = []
        self.callers = []



# read a function code from a c file.
# https://stackoverflow.com/questions/55078713/extract-function-code-from-c-sourcecode-file-with-python
def process_file(filename, line_num):
    if not os.path.exists(filename):
        return ""

    print("opening " + filename + " on line " + str(line_num))

    code = ""
    cnt_braket = 0
    found_start = False
    found_end = False

    # encoding = "ISO-8859-1" for xye server
    with open(filename, "r" ) as f:
        for i, line in enumerate(f):
            if(i >= (line_num - 1)):
                code += line

                if line.count("{") > 0:
                    found_start = True
                    cnt_braket += line.count("{")

                if line.count("}") > 0:
                    cnt_braket -= line.count("}")

                if cnt_braket == 0 and found_start == True:
                    found_end = True
                    print("== len of code: %d" % len(code))
                    return code
    print("== len of code: %d" % len(code))
    return code

def parse_cflow_line(line):
    p1 = re.compile(r'.*?[(][)]', re.S)  # func name
    p2 = re.compile(r'at .*?[.]c:.*?>', re.S)  # filename and locations

    func_name = ""
    file_name  = ""
    file_loc = "0"

    res = re.findall(p1, line)
    if len(res) > 0:
        func_name = res[0].strip()

        res = re.findall(p2, line)
        if len(res) > 0:
            file_name_loc = res[0].strip()[3:-1]
            # print(file_name_loc)
            file_name, file_loc = file_name_loc.split(":")
            # print(file_name, file_loc)

    return func_name, file_name, file_loc

def get_space_num(line):
    n = 0
    for c in line:
        if c==" ":
            n+=1
        else:
            break
    return n



def _recurse_tree(parent, depth, source, c_type=0):
    last_line = source.readline().replace("\t","    ").rstrip()
    while last_line:
        tabs = get_space_num(last_line)
        if tabs < depth:
            break
        # node = last_line.strip()
        func_name, file_name, file_loc = parse_cflow_line(last_line.strip())
        # code 先设置为空，后面再补充
        code = ""
        sub_func = Function(func_name, file_name, file_loc, code)

        if tabs >= depth:
            if parent is not None:
                # print("%s: %s" %(parent, node))
                if c_type==0:
                    parent.callees.append(sub_func)
                else:
                    parent.callers.append(sub_func)
            last_line = _recurse_tree(sub_func, tabs+1, source, c_type)
    return last_line

def recurse_to_dict(func, changed_func_names, depth):
    dd = {
        "func_name": func.func_name,
        "file_name": func.file_name,
        "file_loc": func.file_loc,

        "code": func.code,
        "callers":[],
        "callees":[]
    }
    if len(func.callees) > 0:
        for ff in func.callees:
            # print("==", depth, ff.func_name, changed_func_names)
            if depth == 0 and changed_func_names is not None and ff.file_name!= "" and ff.file_name in changed_func_names.keys() and ff.func_name in changed_func_names[ff.file_name]:
                ff.code = process_file(ff.file_name, int(ff.file_loc))
                dd['callees'].append(recurse_to_dict(ff, changed_func_names, depth + 1))
            if depth > 0:
                ff.code = process_file(ff.file_name, int(ff.file_loc))
                dd['callees'].append(recurse_to_dict(ff, changed_func_names, depth + 1))

    if len(func.callers) > 0:
        for ff in func.callers:
            if depth == 0 and changed_func_names is not None and ff.file_name!= "" and ff.file_name in changed_func_names.keys() and ff.func_name in changed_func_names[ff.file_name]:
                ff.code = process_file(ff.file_name, int(ff.file_loc))
                dd['callers'].append(recurse_to_dict(ff, changed_func_names, depth + 1))
            if depth > 0:
                ff.code = process_file(ff.file_name, int(ff.file_loc))
                dd['callers'].append(recurse_to_dict(ff, changed_func_names, depth + 1))
    return dd

"""
# 【2】列举所有 c 文件 --> 列举所有的文件路径

# 【3】cflow path/*.c

# 【4】找出【1】中的 caller 和 callee

# 【5】然后，最这些 caller 和 callee 提取代码，即为 code_after, caller_after, callee_after
"""
def find_caller_callee(commit, functions, data_type="ffmpeg"):
    logger.info("=== find_caller_callee")
    changed_funcs = {}
    p1 = re.compile(r'[^ ]*?[(]', re.S)
    for ff in functions:
        filename, b = ff.split(":::")
        res = re.findall(p1, b)
        if len(res) > 0:
            func_name = res[0]+")"
            if filename not in changed_funcs.keys():
                changed_funcs[filename] = [func_name]
            else:
                changed_funcs[filename].append(func_name)
    # print("changed_funcs:")
    # print(changed_funcs)



    cmd = 'find . -name "*.c"'
    p = os.popen(cmd)
    x = p.read()

    c_files = x.strip().split("\n")


    # callees
    cmd = "/usr/local/gccgraph/bin/gcc "
    paths = []
    for filename in c_files:
        pp = os.path.dirname(filename)
        if pp not in paths:
            paths.append(pp)
            # print(pp)
            cmd = cmd + pp + "/*.c "

    logger.info("cmd: %s" % cmd)

    p = os.popen(cmd)
    try:
        x = p.read()
    except:
        x = ""
        logger.error("=== p.read() error")


    cmd = 'genfull'
    logger.info("cmd: %s" % cmd)
    p = os.popen(cmd)
    x = p.read()


    return []

def get_all_files_func():
    return 0, 0
    cmd = 'find . -name "*.c"'
    p = os.popen(cmd)
    try:
        x = p.read()
    except:
        x = ""
        logger.error("=== p.read() error")

    c_files = x.strip().split("\n")
    total_c_files = len(c_files)
    total_c_funcs = 0
    for filename in c_files:
        funcs = []

        cmd = "cflow --omit-arguments %s" % filename
        # logger.info("cmd: %s" % cmd)
        p = os.popen(cmd)
        try:
            x = p.read()
        except:
            x = ""
            logger.error("=== p.read() error")


        p1 = re.compile(r'.*?[(][)]', re.S)  # func name
        p2 = re.compile(r'at .*?[.]c:.*?>', re.S)  # filename and locations
        for line in x.strip().split("\n"):
            res = re.findall(p1, line)
            if len(res) > 0:
                func_name = res[0].strip()

                res = re.findall(p2, line)
                if len(res) > 0:
                    file_name = res[0].strip()
                    if func_name not in funcs:
                        funcs.append(func_name)

        total_c_funcs += len(funcs)

    return total_c_files, total_c_funcs

if __name__ == '__main__':
    commit_id_list = []
    code_before_list = []
    caller_before_list = []
    callee_before_list = []
    code_after_list = []
    caller_after_list = []
    callee_after_list = []

    total_c_files_list = []
    total_c_funcs_list = []
    changed_files_num_list = []
    changed_func_num_list = []

    parser = argparse.ArgumentParser(description='Test for argparse')
    parser.add_argument('--log_file', help='log_file', type=str, default='../log/ffmpeg.log')
    args = parser.parse_args()

    pp = os.path.dirname(args.log_file)
    if pp != "." and not os.path.exists(pp):
        os.makedirs(pp)

    if not os.path.exists(json_savepath + "/"):
        os.makedirs(json_savepath + "/")

    if not os.path.exists(show_log_savepath + "/"):
        os.makedirs(show_log_savepath + "/")

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(filename)s line: %(lineno)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=args.log_file)
    logger = logging.getLogger(__name__)

    logger.info("codeviz parameters %s", args)

    df = pd.read_csv(ffmpeg_commits_file)
    df = df.fillna('')
    print(df.head())

    # 提取 每个 vulnerbility 的 commits
    cve_commits = {}
    for index, row in df.iterrows():
        cve_id = row['cve_id']
        commit_id = row['commit_id']
        if cve_id != '':
            if cve_id in cve_commits.keys():
                cve_commits[ cve_id ].append(commit_id)
            else:
                cve_commits[ cve_id ] = [commit_id]

    for cve_id in cve_commits.keys():
        to_json_file = json_savepath + "/%s.json" % cve_id
        if os.path.exists( to_json_file ):
            logger.info("=== %s exists" % to_json_file)
            continue

        print("now:", index, cve_id)
        logger.info("=== now: %d ,cve_id: %s" % (index, cve_id) )

        commits = cve_commits[cve_id]
        commit_functions = []

        for commit in commits:

            # 1. git reset

            # git reset
            cmd = "git reset --hard %s" % commit
            logger.info("cmd: %s" % cmd)
            p = os.popen(cmd)
            try:
                x = p.read()
            except:
                x = ""
                logger.error("=== p.read() error")

            # 统计所有的 file 数量，和 func 数量
            total_c_files, total_c_funcs = get_all_files_func()


            # git show diff
            cmd = "git --no-pager show %s" % commit
            # cmd = " git --no-pager diff %s %s^1" % (commit, affected_tag)

            logger.info("cmd: %s" % cmd)
            p = os.popen(cmd)
            try:
                x = p.read()
            except:
                x = ""
                logger.error("=== p.read() error, cmd: %s" % cmd)


            # to_show_file = "%s/%s_git_diff_.txt" % (show_log_savepath, commit)
            # with open(to_show_file, "w") as f:
            #     f.write(x)
            # print("saved to", to_show_file)
            # logger.info("saved to %s" % to_show_file)


            # 【1】找出 修改的 filename 和 func name （计算 vul， non-vul distribution）
            changed_files_num = 0
            changed_func_num = 0

            changed_func_names = []
            is_c_file = False
            for line in x.strip().split("\n"):
                loc = line.find("+++ b/")
                if loc > -1:
                    filename = line[6:].strip()
                    not_test = True
                    if data_type == "ffmpeg" and (filename.find("/tests/") > -1 or filename.find("/doc/") > -1):
                        not_test = False


                    if not_test and filename[-2:] == '.c':
                        changed_files_num += 1
                        is_c_file = True
                    else:
                        is_c_file = False


                if is_c_file and line.find("@@") > -1:
                    arr = line.strip().split("@@")
                    func_name = "./" + filename + ":::" + arr[-1].strip()
                    # print(arr[-1].strip())
                    if func_name not in changed_func_names:
                        changed_func_names.append(func_name)
                        print("changed func: %s" % func_name )
                        changed_func_num += 1


            functions_after = find_caller_callee(cve_id, changed_func_names)
            exit()

            # git reset 到修复前
            cmd = "git reset --hard %s^1" % commit
            logger.info("cmd: %s" % cmd)
            p = os.popen(cmd)
            try:
                x = p.read()
            except:
                x = ""
                logger.error("=== p.read() error")

            functions_before = find_caller_callee(cve_id, changed_func_names)

            commit_functions.append({
                "commit_id": commit,

                'total_c_files': total_c_files,
                'total_c_funcs': total_c_funcs,
                'changed_files_num': changed_files_num,
                'changed_func_num': changed_func_num,

                "before": functions_before,
                "after": functions_after
            })

        to_data = {
            'cve_id': cve_id,
            'functions': commit_functions,
        }

        # print(json.dumps(to_data))

        with open(to_json_file, "w") as fw:
            fw.write(json.dumps(to_data))
        print("saved to %s" % to_json_file)
        logger.info("saved to %s" % to_json_file)

        # break

        # if index > 100:
        #     break
