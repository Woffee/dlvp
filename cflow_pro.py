"""
May 29, 2021:
    - 大的 project 比如 chrome，linux，php ，因为文件太多，cflow 处理不了。
    - 对于每个 cve，先找出修改了哪些 function，然后，提取出 include 了哪些文件。
    - 然后用 find 命令查找文件，再用 cflow 处理。


May 27, 2021:
    - 整个流程：
        1 先爬所有的 cve 和 对应的 comiit，然后 cve 和 commit 一一对应存起来
        2 获取每一个 project 的 commit 历史，然后对 1 中的结果进行排序
        3 用 cflow 获取每一个 commit 前后的 callee 和 caller


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
import urllib


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
    with open(filename, "r", encoding="utf8", errors='ignore') as f:
        for i, line in enumerate(f):
            if (i >= (line_num - 1)):
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
    file_name = ""
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
        if c == " ":
            n += 1
        else:
            break
    return n


def _recurse_tree(parent, depth, source, c_type=0):
    last_line = source.readline().replace("\t", "    ").rstrip()
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
                if c_type == 0:
                    parent.callees.append(sub_func)
                else:
                    parent.callers.append(sub_func)
            last_line = _recurse_tree(sub_func, tabs + 1, source, c_type)
    return last_line


def recurse_to_dict(func, changed_func_names, depth):
    dd = {
        "func_name": func.func_name,
        "file_name": func.file_name,
        "file_loc": func.file_loc,

        "code": func.code,
        "callers": [],
        "callees": []
    }
    if len(func.callees) > 0:
        for ff in func.callees:
            # print("==", depth, ff.func_name, changed_func_names)
            if depth == 0 and changed_func_names is not None and ff.file_name != "" and ff.file_name in changed_func_names.keys() and ff.func_name in \
                    changed_func_names[ff.file_name]:
                ff.code = process_file(ff.file_name, int(ff.file_loc))
                dd['callees'].append(recurse_to_dict(ff, changed_func_names, depth + 1))
            if depth > 0:
                ff.code = process_file(ff.file_name, int(ff.file_loc))
                dd['callees'].append(recurse_to_dict(ff, changed_func_names, depth + 1))

    if len(func.callers) > 0:
        for ff in func.callers:
            if depth == 0 and changed_func_names is not None and ff.file_name != "" and ff.file_name in changed_func_names.keys() and ff.func_name in \
                    changed_func_names[ff.file_name]:
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


def find_caller_callee(commit, functions, c_files, data_type="ffmpeg"):
    logger.info("=== find_caller_callee")
    changed_funcs = {}
    p1 = re.compile(r'[^ ]*?[(]', re.S)
    for ff in functions:
        filename, b = ff.split(":::")
        res = re.findall(p1, b)
        if len(res) > 0:
            func_name = res[0] + ")"
            if filename not in changed_funcs.keys():
                changed_funcs[filename] = [func_name]
            else:
                changed_funcs[filename].append(func_name)
    # print("changed_funcs:")
    # print(changed_funcs)

    # cmd = 'find . -name "*.c"'
    # p = os.popen(cmd)
    # x = p.read()
    #
    # c_files = x.strip().split("\n")

    # callees
    cmd = "cflow "
    paths = []
    for filename in c_files:
        pp = os.path.dirname(filename)
        if pp not in paths:
            paths.append(pp)
            # print(pp)
            cmd = cmd + pp + "/*.c "

    to_callee_file = "%s/%s_callee.txt" % (show_log_savepath, commit)
    cmd = cmd + " --omit-arguments --depth=3 --all --all"

    logger.info("cmd: %s > %s" % (cmd, to_callee_file))
    p = os.popen(cmd + " > " + to_callee_file)
    x = p.read()

    print("saved to", to_callee_file)

    func_callee = Function("root_callee")
    with open(to_callee_file) as inFile:
        _recurse_tree(func_callee, 0, inFile, 0)

    # callers
    to_caller_file = "%s/%s_caller.txt" % (show_log_savepath, commit)

    cmd += " --reverse "
    logger.info("cmd: %s > %s" % (cmd, to_caller_file))
    p = os.popen(cmd + " > " + to_caller_file)
    x = p.read()

    print("saved to", to_caller_file)
    logger.info("saved to %s" % to_caller_file)

    func_caller = Function("root_caller")
    with open(to_caller_file) as inFile:
        _recurse_tree(func_caller, 0, inFile, 1)

    func_callee_dict = recurse_to_dict(func_callee, changed_funcs, 0)
    func_caller_dict = recurse_to_dict(func_caller, changed_funcs, 0)
    logger.info("=== len of  func_callee_dict['callees']: %d" % len(func_callee_dict['callees']))
    logger.info("=== len of  func_callee_dict['callers']: %d" % len(func_callee_dict['callers']))
    logger.info("=== len of  func_caller_dict['callees']: %d" % len(func_caller_dict['callees']))
    logger.info("=== len of  func_caller_dict['callers']: %d" % len(func_caller_dict['callers']))

    d1 = func_callee_dict['callees']
    d2 = func_caller_dict['callers']

    return d1 + d2


def get_all_files_func():
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


def get_params_from_link(link):
    # print(link)
    url, params_str = link.split("?", 1)
    params = params_str.split("&")

    res = {}
    for p in params:
        k, v = p.split("=")
        res[k] = v
    return res


def get_commits_file(cve_list_file, to_file):
    if not os.path.exists(cve_list_file):
        print("cve_list_file not existed: %s" % cve_list_file)
        logger.error("cve_list_file not existed: %s" % cve_list_file)
        exit()

    log_file = "commit_log.tmp"

    if not os.path.exists(log_file):
        cmd = "git --no-pager log --pretty=oneline --all > %s" % log_file
        p = os.popen(cmd)
        x = p.read()

    all_commits = []
    with open(log_file, "r", encoding="utf8", errors='ignore') as f:
        for line in f.read().strip().split("\n"):
            arr = line.split(" ", 1)
            commit = arr[0]
            all_commits.append(commit)

    df_cve_list = pd.read_csv(cve_list_file)
    df_cve_list = df_cve_list.fillna('')
    print(df_cve_list.head())

    cve_commits = []
    # cve_ref_links = df_cve_list['ref_links']
    commit_cveid = {}

    for index, row in df_cve_list.iterrows():
        cve_id = row['cve_id']

        links = row['ref_links'].split("\n")
        # print(links)
        for link in links:
            cc = re.findall("[a-zA-Z0-9]{40}", link)
            if len(cc) > 0:
                commit_id = cc[0]
                cve_commits.append(commit_id)
                if commit_id in commit_cveid.keys():
                    print("exists: %s" % commit_id)
                    print("before: %s" % commit_cveid[commit_id])
                    print("new:  %s" % cve_id)
                else:
                    commit_cveid[commit_id] = [cve_id, row['affected_tags'], row['ref_links']]

            # if link.find("/git.php.net/") > -1 and link.find("h=") > -1:
            #     # params = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(link.replace(";", "&")).query))
            #     link_ = link.replace(";", "&")
            #     params = get_params_from_link(link_)
            #     commit_id = params['id']
            #     cve_commits.append(commit_id)
            #     if commit_id in commit_cveid.keys():
            #         print("exists: %s" % commit_id)
            #         print("before: %s" % commit_cveid[commit_id])
            #         print("new:  %s" % cve_id)
            #     else:
            #         commit_cveid[commit_id] = [cve_id, row['affected_tags'], row['ref_links']]
            #
            # elif link.find("github.com/") > -1 and link.find("/commit/") > -1:
            #     # print("===github===")
            #     # print(link)
            #     arr = link.split("/")
            #     commit_id = arr[-1]
            #     loc = commit_id.find("#")
            #     if loc > -1:
            #         commit_id = commit_id[:loc]
            #     cve_commits.append(commit_id)
            #     if commit_id in commit_cveid.keys():
            #         print("exists: %s" % commit_id)
            #         print("  before: %s" % commit_cveid[commit_id])
            #         print("  new:  %s" % cve_id)
            #     else:
            #         commit_cveid[commit_id] = [cve_id, row['affected_tags'], row['ref_links']]
            #
            # else:
            #     pass
    print("len(commit_cveid.keys())", len(commit_cveid.keys()))
    print("len(cve_commits): ", len(cve_commits))

    # commit_tag = {}
    # with open("ffmpeg_all_tag_commit_id.txt", "r") as f:
    #     lines = f.read().strip().split("\n")
    #     for l in lines:
    #         commit, tag = l.split(" ", 1)
    #         # print(commit, tag)
    #         commit_tag[commit] = tag
    # print("len(commit_tag.keys()):", len(commit_tag.keys()))

    # all_commits = []
    # with open("FFmpeg_all_branch_commits.txt", "r") as f:
    #     lines = f.read().strip().split("\n")
    #     for l in lines:
    #         commit, _ = l.split(" ", 1)
    #         all_commits.append(commit)
    # print("len of all_commits:", len(all_commits))

    to_commits = []
    to_tags = []
    to_cve_ids = []
    to_affected_tags = []
    to_ref_links = []

    commit_cveid_keys = list(commit_cveid.keys())
    # print(list(commit_cveid.keys()))

    # commit_tag_keys = list(commit_tag.keys())

    for commit in all_commits:
        cve_id = ""
        tag = ""
        affected_tags = ""
        ref_links = ""

        if commit in commit_cveid_keys:
            cve_id, affected_tags, ref_links = commit_cveid[commit]
            # print("111")

        # if commit in commit_tag_keys:
        #     tag = commit_tag[commit]

        if cve_id != "" or tag != "":
            to_commits.append(commit)
            to_tags.append(tag)
            to_cve_ids.append(cve_id)
            to_affected_tags.append(affected_tags)
            to_ref_links.append(ref_links)

    to_data = {
        'commit_id': to_commits,
        'tag': to_tags,
        'cve_id': to_cve_ids,
        'affected_tags': to_affected_tags,
        'ref_links': to_ref_links
    }

    to_df = pd.DataFrame(data=to_data)
    to_df.to_csv(to_file, sep=',', index=None)
    print("saved to: %s" % to_file)
    logger.info("saved to: %s" % to_file)

# 从文件里提取 include 的文件名
def get_includes(filename):
    includes_re = re.compile(r'\s*#\s*include\s+["<](?P<file>.+?)[">]')

    with open(filename, 'rt' , encoding="utf8", errors='ignore') as f:
        data = f.read()

    # Remove all comments
    data = re.sub(r'/\*.*?\*/', '', data, flags=re.DOTALL)
    data = re.sub(r'//.*', '', data)

    includes = includes_re.findall(data)

    return includes

def find_included_files(changed_files):
    res = []
    inc_files = []

    for filename in changed_files:
        includes = get_includes(filename)
        for inc_file in includes:
            inc_file_name = inc_file.split("/")[-1]
            if inc_file_name not in inc_files:
                inc_files.append(inc_file_name)

    for inc_file in inc_files:
        cmd = "find ./ -name %s" % inc_file
        p = os.popen(cmd)
        x = p.read()
        for filename in x.strip().split("\n"):
            if filename not in res:
                res.append(filename)
    return res

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
    parser.add_argument('--data_type', help='data_type', type=str, default='ffmpeg')
    parser.add_argument('--log_file', help='log_file', type=str, default='../log/ffmpeg.log')
    args = parser.parse_args()

    data_path = "../data2"
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    data_type = args.data_type

    ffmpeg_commits_file = data_path + "/%s_commits_in_order.csv" % data_type
    cve_list_file = data_path + "/%s_vul_list.csv" % data_type
    # df_file = data_path + "/FFmpeg.csv"

    to_df_file = "%s_caller_callee.csv" % data_type

    pp = os.path.dirname(args.log_file)
    if pp != "." and not os.path.exists(pp):
        os.makedirs(pp)

    json_savepath = data_path + "/%s_jsons" % data_type
    if not os.path.exists(json_savepath + "/"):
        os.makedirs(json_savepath + "/")

    show_log_savepath = data_path + "/%s_tmp" % data_type
    if not os.path.exists(show_log_savepath + "/"):
        os.makedirs(show_log_savepath + "/")

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(filename)s line: %(lineno)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=args.log_file)
    logger = logging.getLogger(__name__)

    logger.info("cflow parameters %s", args)

    if not os.path.exists(ffmpeg_commits_file):
        get_commits_file(cve_list_file, ffmpeg_commits_file)

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
                cve_commits[cve_id].append(commit_id)
            else:
                cve_commits[cve_id] = [commit_id]

    ii = -1
    for cve_id in cve_commits.keys():
        ii += 1

        to_json_file = json_savepath + "/%s.json" % cve_id
        if os.path.exists(to_json_file):
            logger.info("=== %s exists" % to_json_file)
            continue

        print("now:", ii, cve_id)
        logger.info("=== now: %d ,cve_id: %s" % (index, cve_id))

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
            # total_c_files, total_c_funcs = get_all_files_func()
            total_c_files = 0
            total_c_funcs = 0

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
            changed_file_names = []
            is_c_file = False
            for line in x.strip().split("\n"):
                loc = line.find("+++ b/")
                if loc > -1:
                    filename = line[6:].strip()
                    not_test = True
                    if (filename.find("/tests/") > -1 or filename.find("/doc/") > -1):
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
                        print("changed func: %s" % func_name)
                        changed_func_num += 1
                    if filename not in changed_file_names:
                        changed_file_names.append(filename)

            included_files = find_included_files(changed_file_names)

            functions_after = find_caller_callee(cve_id, changed_func_names, included_files)

            # git reset 到修复前
            cmd = "git reset --hard %s^1" % commit
            logger.info("cmd: %s" % cmd)
            p = os.popen(cmd)
            try:
                x = p.read()
            except:
                x = ""
                logger.error("=== p.read() error")

            functions_before = find_caller_callee(cve_id, changed_func_names, included_files)

            commit_functions.append({
                "commit_id": commit,

                'total_c_files': total_c_files,
                'total_c_funcs': total_c_funcs,
                'changed_files_num': changed_files_num,
                'changed_func_num': changed_func_num,

                "functions_before": functions_before,
                "functions_after": functions_after
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
    print("done")
    logger.info("done")