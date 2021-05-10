"""
git --no-pager show e43a0a232dbf6d3c161823c2e07c52e76227a1bc

cflow --main who_am_i whoami.c

cflow --main=filter_frame --depth=2 libavfilter/vf_showinfo.c




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



data_path = "../data"
ffmpeg_commits_file = data_path + "/FFmpeg_commits_in_order.txt"
df_file = data_path + "/FFmpeg.csv"
show_log_savepath = data_path + "/FFmpeg"
vul_distribution_file = data_path + "FFmpeg_vul_distributions.csv"
to_df_file = "FFmpeg_caller_callee.csv"




# read a function code from a c file.
# https://stackoverflow.com/questions/55078713/extract-function-code-from-c-sourcecode-file-with-python
def process_file(filename, line_num):
    print("opening " + filename + " on line " + str(line_num))

    code = ""
    cnt_braket = 0
    found_start = False
    found_end = False

    with open(filename, "r") as f:
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

"""
# 【2】列举所有 c 文件 --> 列举所有的文件路径

# 【3】cflow path/*.c

# 【4】找出【1】中的 caller 和 callee

# 【5】然后，最这些 caller 和 callee 提取代码，即为 code_after, caller_after, callee_after
"""
def find_caller_callee(commit, functions):



    cmd = 'find . -name "*.c"'
    p = os.popen(cmd)
    x = p.read()

    c_files = x.strip().split("\n")
    cmd = "cflow --omit-arguments "
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

    for line in x.strip().split("\n"):
        func_name, file_name, file_loc = parse_cflow_line(line)
        cur_name = file_name + ":::" + func_name
        print("cur name: %s" % cur_name)
        sp_num = get_space_num(line)

        if flag_start and sp_num > flag_start_sp_n:
            if not cur_name in visited_callee:
                visited_callee.append(cur_name)
                if file_name != "":
                    callee = process_file(file_name, int(file_loc))
                    callee_list.append(callee)
        else:
            flag_start = False
            if cur_name in functions and not cur_name in visited:
                code = process_file(file_name, int(file_loc))
                code_list.append(code)
                visited.append(cur_name)
                flag_start = True
                flag_start_sp_n = sp_num

    cmd += " --reverse"
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
            flag_start = False
            if cur_name in functions and not cur_name in visited:
                visited.append(cur_name)
                flag_start = True
                flag_start_sp_n = sp_num

    code = "\n---code---\n".join(code_list)
    callee = "\n----callee----\n".join(callee_list)
    caller = "\n----caller----\n".join(caller_list)
    return code, callee, caller

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
    parser.add_argument('--log_file', help='log_file', type=str, default='cflow.log')
    args = parser.parse_args()

    pp = os.path.dirname(args.log_file)
    if pp != "." and not os.path.exists(pp):
        os.makedirs(pp)

    if not os.path.exists(show_log_savepath + "/"):
        os.makedirs(show_log_savepath + "/")

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(filename)s line: %(lineno)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=args.log_file)
    logger = logging.getLogger(__name__)

    logger.info("cflow parameters %s", args)

    commits_in_order = []
    with open(ffmpeg_commits_file, "r") as f:
        for commit in f.read().strip().split("\n"):
            commits_in_order.append(commit)

    # 1. git reset
    for commit in commits_in_order:
        logger.info("=== start, commit %s" % commit)
        # git reset
        cmd = "git reset --hard %s" % commit
        p = os.popen(cmd)
        x = p.read()

        # 统计所有的 file 数量，和 func 数量
        total_c_files, total_c_funcs = get_all_files_func()


        # git show
        cmd = "git --no-pager show %s" % commit
        p = os.popen(cmd)
        x = p.read()
        to_show_file = "%s/%s_git_show.txt" % (show_log_savepath, commit)
        with open(to_show_file, "w") as f:
            f.write(x)
        print("saved to", to_show_file)
        logger.info("saved to %s" % to_show_file)


        # 【1】找出 修改的 filename 和 func name，顺便计算 vul， non-vul distribution
        changed_files_num = 0
        changed_func_num = 0

        changed_func_names = []
        for line in x.strip().split("\n"):
            loc = line.find("+++ b/")
            if loc > -1:
                filename = line[6:].strip()
                changed_files_num += 1

            if line.find("@@") > -1:
                arr = line.strip().split("@@")
                func_name = "./" + filename + ":::" + arr[-1].strip()
                # print(arr[-1].strip())
                if func_name not in changed_func_names:
                    changed_func_names.append(func_name)
                    print("changed func: %s" % func_name )
                    changed_func_num += 1

        commit_id_list.append(commit)
        total_c_files_list.append(total_c_files)
        total_c_funcs_list.append(total_c_funcs)
        changed_files_num_list.append(changed_files_num)
        changed_func_num_list.append(changed_func_num)

        # vul_distributions.append( [total_c_files, total_c_funcs, changed_files_num, changed_func_num] )

        code, callee, caller = find_caller_callee(commit, changed_func_names)
        code_after_list.append(code)
        callee_after_list.append(callee)
        caller_after_list.append(caller)


        # git reset 到本次修改前

        # 重复 【2】- 【5】，即为 code_before, caller_before, callee_before
        code_before_list.append("")
        callee_before_list.append("")
        caller_before_list.append("")

        logger.info("=== done, commit %s" % commit)
        break

    to_data = {
        'commit_id': commit_id_list,

        'total_c_files': total_c_files_list,
        'total_c_funcs': total_c_funcs_list,
        'changed_files_num': changed_files_num_list,
        'changed_func_num': changed_func_num_list,

        'code_after': code_after_list,
        'callee_after': callee_after_list,
        'caller_after': caller_after_list,

        'code_before': code_before_list,
        'callee_before': callee_before_list,
        'caller_before': caller_before_list,
    }
    to_df = pd.DataFrame(data=to_data)
    to_df.to_csv(to_df_file, sep=',', index=None)

