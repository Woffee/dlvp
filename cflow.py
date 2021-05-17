"""

 -- 假设：每一个 版本号 修复一个 vulnerability --



git --no-pager show e43a0a232dbf6d3c161823c2e07c52e76227a1bc

cflow --main who_am_i whoami.c

cflow --main=filter_frame --depth=2 libavfilter/vf_showinfo.c

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

data_path = "../data2"
ffmpeg_commits_file = data_path + "/ffmpeg_commits_in_order_with_fixed_tag.csv"
# df_file = data_path + "/FFmpeg.csv"

show_log_savepath = data_path + "/FFmpeg"
vul_distribution_file = data_path + "FFmpeg_vul_distributions.csv"
to_df_file = "FFmpeg_caller_callee.csv"

json_savepath = data_path + "/FFmpeg_jsons"


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

    with open(filename, "r") as f:
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


"""
# 【2】列举所有 c 文件 --> 列举所有的文件路径

# 【3】cflow path/*.c

# 【4】找出【1】中的 caller 和 callee

# 【5】然后，最这些 caller 和 callee 提取代码，即为 code_after, caller_after, callee_after
"""


def find_caller_callee(commit, functions):
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

    cmd = 'find . -name "*.c"'
    p = os.popen(cmd)
    x = p.read()

    c_files = x.strip().split("\n")
    cmd = "cflow --omit-arguments --no-main --depth=2"
    paths = []
    for filename in c_files:
        pp = os.path.dirname(filename)
        if pp not in paths:
            paths.append(pp)
            # print(pp)
            cmd = cmd + pp + "/*.c "
    p = os.popen(cmd)
    x = p.read()

    to_file = "%s/%s_callee.txt" % (show_log_savepath, commit)
    with open(to_file, "w") as fw:
        fw.write(x)
    print("saved to", to_file)
    logger.info("saved to %s" % to_file)

    # find callee
    code_list = []
    callee_list = []
    visited = []
    visited_callee = []

    flag_start = False
    flag_start_sp_n = 0

    function_list = []
    function_detail = {}

    for line in x.strip().split("\n"):
        func_name, file_name, file_loc = parse_cflow_line(line)
        cur_name = file_name + ":::" + func_name
        # print("cur name: %s" % cur_name)
        sp_num = get_space_num(line)

        if flag_start and sp_num > flag_start_sp_n:
            if not cur_name in visited_callee:
                visited_callee.append(cur_name)
                if file_name != "":
                    callee = process_file(file_name, int(file_loc))
                    callee_list.append(callee)
        else:
            if len(function_detail.keys()) > 0:
                function_detail['callees'] = callee_list
                function_list.append(function_detail)

            flag_start = False
            function_detail = {}
            callee_list = []

            if file_name != "" and file_name in changed_funcs.keys() and func_name in changed_funcs[
                file_name] and not cur_name in visited:
                print("== bingo")
                code = process_file(file_name, int(file_loc))
                function_detail = {
                    'file_name': file_name,
                    'file_loc': file_loc,
                    'func_name': func_name,
                    'code': code,
                    'callers': [],
                    'callees': []
                }
                # code_list.append(code)
                visited.append(cur_name)
                flag_start = True
                flag_start_sp_n = sp_num

    cmd += " --reverse "
    p = os.popen(cmd)
    x = p.read()
    to_file = "%s/%s_caller.txt" % (show_log_savepath, commit)
    with open(to_file, "w") as fw:
        fw.write(x)
    print("saved to", to_file)
    logger.info("saved to %s" % to_file)

    caller_list = []
    visited = []
    visited_caller = []
    flag_start = False
    flag_start_sp_n = 0

    function_detail = {}

    for line in x.strip().split("\n"):
        func_name, file_name, file_loc = parse_cflow_line(line)
        cur_name = file_name + ":::" + func_name
        sp_num = get_space_num(line)

        if flag_start and sp_num > flag_start_sp_n:
            if not cur_name in visited_caller:
                visited_caller.append(cur_name)
                if file_name != "":
                    caller = process_file(file_name, int(file_loc))
                    caller_list.append(caller)
        else:
            if len(function_detail.keys()) > 0:
                function_detail['callers'] = caller_list
                function_list.append(function_detail)

            flag_start = False
            function_detail = {}
            callee_list = []

            if file_name != "" and file_name in changed_funcs.keys() and func_name in changed_funcs[
                file_name] and not cur_name in visited:
                print("== bingo caller, cur_name: %s" % cur_name)
                visited.append(cur_name)
                flag_start = True
                flag_start_sp_n = sp_num

                code = process_file(file_name, int(file_loc))
                function_detail = {
                    'file_name': file_name,
                    'file_loc': file_loc,
                    'func_name': func_name,
                    'code': code,
                    'callers': [],
                    'callees': []
                }

    # code = "\n---code---\n".join(code_list)
    # callee = "\n----callee----\n".join(callee_list)
    # caller = "\n----caller----\n".join(caller_list)
    # return code, callee, caller

    return function_list


def get_all_files_func():
    cmd = 'find . -name "*.c"'
    p = os.popen(cmd)
    x = p.read()

    c_files = x.strip().split("\n")
    total_c_files = len(c_files)
    total_c_funcs = 0
    for filename in c_files:
        funcs = []

        cmd = "cflow --omit-arguments %s" % filename
        p = os.popen(cmd)
        x = p.read()

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

    logger.info("cflow parameters %s", args)

    df = pd.read_csv(ffmpeg_commits_file)
    df = df.fillna('')
    print(df.head())

    for index, row in df.iterrows():

        if row['cve_id'] == '':
            continue

        print("now:", index, row['cve_id'])
        logger.info("=== now: %d ,cve_id: %s" % (index, row['cve_id']))

        cve_id = row['cve_id']
        fixed_tag = row['fixed_tag']
        affected_tag = row['affected_tag']

        # 1. git reset

        # git reset
        cmd = "git reset --hard %s" % fixed_tag
        p = os.popen(cmd)
        x = p.read()

        # 统计所有的 file 数量，和 func 数量
        total_c_files, total_c_funcs = get_all_files_func()

        # git show diff
        # cmd = "git --no-pager show %s" % commit
        cmd = " git --no-pager diff %s %s" % (fixed_tag, affected_tag)
        p = os.popen(cmd)
        x = p.read()
        to_show_file = "%s/%s_git_diff_.txt" % (show_log_savepath, fixed_tag)
        with open(to_show_file, "w") as f:
            f.write(x)
        print("saved to", to_show_file)
        logger.info("saved to %s" % to_show_file)

        # 【1】找出 修改的 filename 和 func name，顺便计算 vul， non-vul distribution
        changed_files_num = 0
        changed_func_num = 0

        changed_func_names = []
        is_c_file = False
        for line in x.strip().split("\n"):
            loc = line.find("+++ b/")
            if loc > -1:
                filename = line[6:].strip()
                if filename[-2:] == '.c':
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

        functions_after = find_caller_callee(cve_id, changed_func_names)

        # git reset 到修复前
        cmd = "git reset --hard %s" % affected_tag
        p = os.popen(cmd)
        x = p.read()

        functions_before = find_caller_callee(cve_id, changed_func_names)

        to_data = {
            'cve_id': cve_id,
            'affected_tag': affected_tag,
            'fixed_tag': fixed_tag,

            'total_c_files': total_c_files,
            'total_c_funcs': total_c_funcs,
            'changed_files_num': changed_files_num,
            'changed_func_num': changed_func_num,

            'functions_after': functions_after,
            'functions_before': functions_before

            # 'code_after': code_after_list,
            # 'callee_after': callee_after_list,
            # 'caller_after': caller_after_list,

            # 'code_before': code_before_list,
            # 'callee_before': callee_before_list,
            # 'caller_before': caller_before_list,
        }

        # print(json.dumps(to_data))
        to_json_file = json_savepath + "/%s.json" % cve_id
        with open(to_json_file, "w") as fw:
            fw.write(json.dumps(to_data))
        print("saved to %s" % to_json_file)
        logger.info("saved to %s" % to_json_file)

        # if index > 100:
        #     break

        # commit_id_list.append(commit)
        # total_c_files_list.append(total_c_files)
        # total_c_funcs_list.append(total_c_funcs)
        # changed_files_num_list.append(changed_files_num)
        # changed_func_num_list.append(changed_func_num)

        # vul_distributions.append( [total_c_files, total_c_funcs, changed_files_num, changed_func_num] )

        # code, callee, caller = find_caller_callee(commit, changed_func_names)
        # code_after_list.append(code)
        # callee_after_list.append(callee)
        # caller_after_list.append(caller)

        # 重复 【2】- 【5】，即为 code_before, caller_before, callee_before
        # code, callee, caller = find_caller_callee(commit+"_before", changed_func_names)
        # code_before_list.append(code)
        # callee_before_list.append(callee)
        # caller_before_list.append(caller)

        # logger.info("=== done, commit %s" % commit)

    # to_data = {
    #     'commit_id': commit_id_list,

    #     'total_c_files': total_c_files_list,
    #     'total_c_funcs': total_c_funcs_list,
    #     'changed_files_num': changed_files_num_list,
    #     'changed_func_num': changed_func_num_list,

    #     'code_after': code_after_list,
    #     'callee_after': callee_after_list,
    #     'caller_after': caller_after_list,

    #     'code_before': code_before_list,
    #     'callee_before': callee_before_list,
    #     'caller_before': caller_before_list,
    # }
    # to_df = pd.DataFrame(data=to_data)
    # to_df.to_csv(to_df_file, sep=',', index=None)

