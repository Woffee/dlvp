"""
和 jiahao 的数据做对比，看增加了多少。

"""


import os
import json
import pandas as pd

base = "./data/jsons"
to_file = "wenbo_data.csv"

project_list = []
cve_id_list = []

func_name_list = []
file_name_list = []

callers_total_before_list = []
callees_total_before_list = []

callers_total_after_list = []
callees_total_after_list = []

def get_project_table(filepath):
    res = {}
    for f1 in os.listdir(filepath):
        for f2 in os.listdir(filepath + "/" + f1):
            ff = f1[:-1]
            res[ff] = f2
    return res

projects_table = get_project_table("projects")

def get_callee_num(func):
    res = 0
    k = 'callees'
    res += len(func[k])
    if len(func[k]) > 0:
        for sub_func in func[k]:
            res += get_caller_num(sub_func)

    return res

def get_caller_num(func):
    res = 0
    k = 'callers'
    res += len(func[k])
    if len(func[k]) > 0:
        for sub_func in func[k]:
            res += get_caller_num(sub_func)

    return res


for root, ds, fs in os.walk(base):
    for f in fs:
        fullname = os.path.join(root, f)
        print(fullname)

        name_arr = fullname.split("/")
        project_name = name_arr[3]
        pos = project_name.find("_jsons")
        if pos > -1:
            project_name = project_name[:pos]

        if project_name in projects_table.keys():
            project_name = projects_table[project_name]

        with open(fullname, "r") as fr:
            txt = fr.read()
        cve = json.loads(txt)



        total_callers_before = 0
        total_callers_after = 0
        total_callees_before = 0
        total_callees_after = 0

        func_stats = {}

        for func in cve['functions']:
            for ff in func['functions_before']:
                func_name = ff['func_name']
                file_name = ff['file_name']
                callee_num1 = get_callee_num(ff)
                caller_num1 = get_caller_num(ff)

                if func_name not in func_stats.keys():
                    func_stats[ func_name ] = {
                        'file_name': file_name,
                        'callee_num1': callee_num1,
                        'caller_num1': caller_num1,
                        'callee_num2': 0,
                        'caller_num2': 0,
                    }
                else:
                    func_stats[func_name]['callee_num1'] += callee_num1
                    func_stats[func_name]['caller_num1'] += caller_num1

            for ff in func['functions_after']:
                func_name = ff['func_name']
                file_name = ff['file_name']
                callee_num2 = get_callee_num(ff)
                caller_num2 = get_caller_num(ff)

                if func_name not in func_stats.keys():
                    func_stats[ func_name ] = {
                        'file_name': file_name,
                        'callee_num1': 0,
                        'caller_num1': 0,
                        'callee_num2': callee_num2,
                        'caller_num2': caller_num2,
                    }
                else:
                    func_stats[func_name]['callee_num2'] += callee_num2
                    func_stats[func_name]['caller_num2'] += caller_num2



        for fff in func_stats.keys():
            v = func_stats[fff]

            project_list.append(project_name)
            cve_id_list.append(cve['cve_id'])
            func_name_list.append(fff)
            file_name_list.append(v['file_name'])
            callers_total_before_list.append(v['caller_num1'])
            callers_total_after_list.append(v['caller_num2'])
            callees_total_before_list.append(v['callee_num1'])
            callees_total_after_list.append(v['callee_num2'])

to_data = {
    'project': project_list,
    'cve_id': cve_id_list,
    'func_name': func_name_list,
    'file_name': file_name_list,
    'callers_total_before': callers_total_before_list,
    'callers_total_after': callers_total_after_list,
    'callees_total_before': callees_total_before_list,
    'callees_total_after': callees_total_after_list,
}


to_df = pd.DataFrame(data=to_data)
to_df.to_csv(to_file, sep=',', index=None)
print("saved to", to_file)


