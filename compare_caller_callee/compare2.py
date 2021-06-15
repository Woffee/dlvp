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
            'func_name': row['func_name'],
            'file_name': row['file_name'],
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





if __name__ == '__main__':
    jiahao_file = "jiahao_data_with_func_name_no_code.csv"
    to_jiahao_file = "jiahao_data_project_level.csv"

    jiahao_data = read_jiahao_data(jiahao_file)
    if not os.path.exists(to_jiahao_file):
        to_jiahao_data(jiahao_data, to_jiahao_file)
    pass