"""


@Time    : 6/15/21
@Author  : Wenbo
"""


import pandas as pd
import os

def read_jiahao_data(filename):
    df = pd.read_csv(filename)
    df = df.fillna('')
    projects = {}
    for index, row in df.iterrows():
        proj = row['project'].lower().strip()
        cve_id = row['CVE ID'].strip()
        if proj == "" or cve_id == "":
            continue

        dd = {
            'func_name': row['func_name'].strip()+"()",
            'file_name': "./" + row['file_name'].strip(),
            'commit_id': row['commit_id'],
            'callee_num': int(row['callee_num']) if row['callee_num'] != '' else 0,
            'caller_num': int(row['caller_num']) if row['caller_num'] != '' else 0,
        }

        if proj not in projects.keys():
            projects[proj] = {}

        if cve_id not in projects[proj].keys():
            projects[proj][cve_id] = []

        projects[proj][cve_id].append(dd)

    return projects

def to_jiahao_data(jiahao_data, to_file):
    project_list = []
    cve_count_all_list = []
    cve_count_cc_list = []
    total_changed_functions_list = []
    total_changed_functions_cc_list = []
    total_callers_list = []
    total_callees_list = []
    for p in jiahao_data.keys():
        cves = jiahao_data[p]

        cve_count_all = len(cves.keys())
        cve_count_cc = 0

        total_changed_functions_all = 0
        total_changed_functions_cc = 0

        total_callers = 0
        total_callees = 0

        for cve_id in cves.keys():
            functions = cves[cve_id]

            total_changed_functions_all += len(functions)

            cve_has_cc = False
            for func in functions:
                total_callers += func['caller_num']
                total_callees += func['callee_num']
                if func['caller_num'] > 0 or func['callee_num'] > 0:
                    cve_has_cc = True
                    total_changed_functions_cc += 1
            if cve_has_cc:
                cve_count_cc += 1


        project_list.append(p)
        cve_count_all_list.append(cve_count_all)
        cve_count_cc_list.append(cve_count_cc)

        total_changed_functions_list.append(total_changed_functions_all)
        total_changed_functions_cc_list.append(total_changed_functions_cc)

        total_callers_list.append(total_callers)
        total_callees_list.append(total_callees)

    df = pd.DataFrame({
        'project': project_list,
        'cve_count(has commits)': cve_count_all_list,
        'cve_count(has callers or callees)': cve_count_cc_list,
        'total_functions(all changed)': total_changed_functions_list,
        'total_functions(has callers or callees)': total_changed_functions_cc_list,
        'total_callers': total_callers_list,
        'total_callees': total_callees_list
    })
    df.to_csv(to_file, index=False)
    print("saved to %s" % to_file)


def read_wenbo_data(filename):
    df = pd.read_csv(filename)
    df = df.fillna('')
    projects = {}
    for index, row in df.iterrows():
        proj = row['project'].lower().strip()
        cve_id = row['cve_id'].strip()
        if proj == "" or cve_id == "":
            continue

        dd = {
            'func_name': row['func_name'],
            'file_name': row['file_name'],
            'commit_id': row['commit_id'],
            'callee_num': int(row['callees_total_before']) if row['callees_total_before'] != '' else 0,
            'caller_num': int(row['callers_total_before']) if row['callers_total_before'] != '' else 0,
        }

        if proj not in projects.keys():
            projects[proj] = {}

        if cve_id not in projects[proj].keys():
            projects[proj][cve_id] = []

        projects[proj][cve_id].append(dd)

    return projects


def compare(jh_data, wb_data, to_file, jiahao_succ_cve_id_list):
    project_list = []
    cve_count_all_list = []
    cve_count_cc_list = []
    total_changed_functions_list = []
    total_changed_functions_cc_list = []
    total_callers_list = []
    total_callees_list = []
    total_commits_list = []

    is_new_project_list = []
    new_functions_list = []
    new_functions_cc_list = []
    new_cves_list = []
    new_cves_cc_list = []
    new_commits_list = []


    for p in wb_data.keys():
        cves = wb_data[p]

        cve_count_all = len(cves.keys())
        cve_count_cc = 0

        total_changed_functions_all = 0
        total_changed_functions_cc = 0

        total_callers = 0
        total_callees = 0
        total_commits = 0


        new_funcs_count = 0    # new functions
        new_funcs_cc_count = 0 # new functions and has callers or callees

        new_cves_count = 0    # new cves
        new_cves_cc_count = 0 # new cves and has callers or callees

        new_commits_count = 0

        # 判断是否是新 project
        is_new_proj = 0
        if p not in jh_data.keys():
            is_new_proj = 1

        for cve in cves.keys():
            # 判断是否是新 cve
            is_new_cve = 0
            if is_new_proj:
                is_new_cve = 1
            else:
                if cve not in jiahao_succ_cve_id_list:
                    is_new_cve = 1

            jh_func_names = []
            jh_commits = []
            if not is_new_cve and p in jh_data.keys() and cve in jh_data[p].keys():
                jh_funcs = jh_data[p][cve]
                for ff in jh_funcs:
                    if ff['func_name'] not in jh_func_names:
                        jh_func_names.append(ff['func_name'])
                    if ff['commit_id'] not in jh_commits:
                        jh_commits.append(ff['commit_id'])


            functions = cves[cve]

            total_changed_functions_all += len(functions)

            cve_has_cc = False
            commits_visted = []
            for func in functions:
                # 判读是否是 新 function
                is_new_func = 0
                if is_new_proj or is_new_cve:
                    is_new_func = 1
                if not is_new_cve and func['func_name'] not in jh_func_names:
                    is_new_func = 1

                total_callers += func['caller_num']
                total_callees += func['callee_num']
                if func['caller_num'] > 0 or func['callee_num'] > 0:
                    cve_has_cc = True
                    total_changed_functions_cc += 1
                    if is_new_func:
                        new_funcs_cc_count += 1

                # 判断是否是新 commit
                if not is_new_cve and func['commit_id'] not in commits_visted and func['commit_id'] not in jh_commits:
                    new_commits_count += 1
                    commits_visted.append(func['commit_id'])

                if is_new_cve and func['commit_id'] not in commits_visted:
                    new_commits_count += 1
                    commits_visted.append(func['commit_id'])

                if func['commit_id'] not in commits_visted:
                    commits_visted.append( func['commit_id'] )

                if is_new_func:
                    new_funcs_count += 1


            if cve_has_cc:
                cve_count_cc += 1
            if is_new_cve:
                new_cves_count += 1
            if cve_has_cc and is_new_cve:
                new_cves_cc_count += 1
            total_commits += len(commits_visted)

        project_list.append(p)
        cve_count_all_list.append(cve_count_all)
        cve_count_cc_list.append(cve_count_cc)

        total_changed_functions_list.append(total_changed_functions_all)
        total_changed_functions_cc_list.append(total_changed_functions_cc)

        total_callers_list.append(total_callers)
        total_callees_list.append(total_callees)
        total_commits_list.append(total_commits)

        is_new_project_list.append(is_new_proj)
        new_functions_list.append(new_funcs_count)
        new_functions_cc_list.append(new_funcs_cc_count)

        new_cves_list.append(new_cves_count)
        new_cves_cc_list.append(new_cves_cc_count)

        new_commits_list.append(new_commits_count)

    df = pd.DataFrame({
        'project': project_list,
        'cve_count(has commits)': cve_count_all_list,
        'cve_count(has callers or callees)': cve_count_cc_list,
        'total_functions(all changed)': total_changed_functions_list,
        'total_functions(has callers or callees)': total_changed_functions_cc_list,
        'total_callers': total_callers_list,
        'total_callees': total_callees_list,
        'total_commits': total_commits_list,

        'is_new_project': is_new_project_list,
        'new_functions': new_functions_list,
        'new_functions(has callers or callees)': new_functions_cc_list,
        'new_cves': new_cves_list,
        'new_cves(has callers or callees)': new_cves_cc_list,
        'new_commits': new_commits_list,
    })
    df.to_csv(to_file, index=False)
    print("saved to %s" % to_file)


if __name__ == '__main__':
    jiahao_file = "jiahao_data_with_func_name_no_code.csv"
    to_jiahao_file = "jiahao_data_project_level.csv"

    wenbo_file = "wenbo_data.csv"
    to_wenbo_file = "wenbo_data_project_level.csv"

    # df = pd.read_csv(wenbo_file)
    # print(len(df['cve_id'].unique()))
    # exit()

    all_jiahao_cves = []
    df1 = pd.read_csv(jiahao_file)
    df1_succ = df1[(df1['callee_num'] > 0) | (df1['caller_num'] > 0)]
    jiahao_succ_cve_id_list = list(df1_succ['CVE ID'].unique())

    jiahao_data = read_jiahao_data(jiahao_file)
    if not os.path.exists(to_jiahao_file):
        to_jiahao_data(jiahao_data, to_jiahao_file)

    wenbo_data = read_wenbo_data(wenbo_file)
    compare(jiahao_data, wenbo_data, to_wenbo_file, jiahao_succ_cve_id_list)
    pass