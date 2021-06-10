"""
May 22, 2021:
    - 这个文件放到 data2 文件夹下
    - 每个 callee 都是一个 function object，需要递归计算个数。

May 16, 2021:
    - 遍历所有的 json 文件，统计每个 cve_id 的记录


May 17, 2021:
    - cve_id, changed_files_num, changed_func_num, , func_name, callers_num_before, callers_num_after, callees_num_before, callees_num_after

{
    "cve_id":"CVE-2019-17542",
    "functions":[
        {
            "commit_id":"02f909dc24b1f05cfbba75077c7707b905e63cd2",
            "total_c_files":2622,
            "total_c_funcs":21139,
            "changed_files_num":1,
            "changed_func_num":1,
            "before":[
                {
                    "file_name":"./libavcodec/vqavideo.c",
                    "file_loc":"121",
                    "func_name":"vqa_decode_init()",
                    "code": ""
                    "callers": [...],
                    "callees": [...],
                }]
            "after":[...]
        }
    ]
}
"""


import os
import json
import pandas as pd

base = "./FFmpeg_jsons"
to_file = "ffmpeg_caller_callee_stats_commits_each_vul_depth4.csv"

cve_id_list = []
changed_files_num_list = []
changed_func_num_list = []

callers_total_before_list = []
callers_total_after_list = []
callees_total_before_list = []
callees_total_after_list = []

# 下面这列 放到一个 单元格里，用 换行符 隔开
func_name_list = []



for root, ds, fs in os.walk(base):
    for f in fs:
        file_path = base + "/" + f
        print(f)
        with open(file_path, "r") as fr:
            txt = fr.read()
        cve = json.loads(txt)



        tmp_files_names = []
        tmp_func_names = []

        total_callers_before = 0
        total_callers_after = 0
        total_callees_before = 0
        total_callees_after = 0

        for func in cve['functions']:
            for ff in func['before']:
                full_func_name = ff['file_name'] + ":::" + ff['func_name']
                if ff['file_name'] not in tmp_files_names:
                    tmp_files_names.append( ff['file_name'] )
                if full_func_name not in tmp_func_names:
                    tmp_func_names.append( full_func_name)

                total_callers_before += len(ff['callers'])
                total_callees_before += len(ff['callees'])

            for ff in func['after']:
                if ff['file_name'] not in tmp_files_names:
                    tmp_files_names.append( ff['file_name'] )
                if full_func_name not in tmp_func_names:
                    tmp_func_names.append( full_func_name)

                total_callers_after += len(ff['callers'])
                total_callees_after += len(ff['callees'])



        cve_id_list.append( cve['cve_id'] )
        changed_files_num_list.append( len(tmp_files_names) )
        changed_func_num_list.append( len(tmp_func_names) )

        callers_total_before_list.append( total_callers_before )
        callers_total_after_list.append( total_callers_after )

        callees_total_before_list.append( total_callees_before )
        callees_total_after_list.append( total_callees_after )

        func_name_list.append("\n".join(tmp_func_names) )



to_data = {
    'cve_id': cve_id_list,
    'changed_files_num': changed_files_num_list,
    'changed_func_num': changed_func_num_list,

    'callers_total_before': callers_total_before_list,
    'callers_total_after': callers_total_after_list,
    'callees_total_before': callees_total_before_list,
    'callees_total_after': callees_total_after_list,

    'changed_func_names': func_name_list

}


to_df = pd.DataFrame(data=to_data)
to_df.to_csv(to_file, sep=',', index=None)
print("saved to", to_file)


