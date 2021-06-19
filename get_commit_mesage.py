"""
遍历每一个 project 的 commits 记录，补充到 wenbo_data.csv 文件里.

@Time    : 6/18/21
@Author  : Wenbo
"""
import pandas as pd

commit_logs_file = "commit_log_files.txt"
to_file = "wenbo_data_with_commit_message.csv"

def get_log_data(log_file):
    data = {}
    with open(log_file, "r", encoding="utf8", errors='ignore') as f:
        for line in f.readlines():
            l = line.strip()
            if l=="":
                continue
            arr = l.split(" ", 1)
            if len(arr) > 1:
                data[arr[0]] = arr[1]
            else:
                data[arr[0]] = ""
    return data

if __name__ == '__main__':
    log_files = {}
    with open(commit_logs_file) as f:
        for line in f.readlines():
            l = line.strip()
            if l=="":
                continue
            arr = l.split("/")

            k = arr[2]
            if k in log_files:
                print(k, log_files[k])
                print(k, l)

            log_files[k] = l

    df_wb_data = pd.read_csv("wenbo_data.csv")
    df_wb_data['commit_message'] = ""

    d_index = list(df_wb_data.columns).index('commit_message')

    for index, row in df_wb_data.iterrows():
        p = row['project']
        if p not in log_files.keys():
            continue

        print("now: %d" % index)

        log_data = get_log_data(log_files[p])
        commit_message = ""
        if row['commit_id'] in log_data.keys():
            commit_message = log_data[row['commit_id']]
        df_wb_data.iloc[index, d_index] = commit_message

    df_wb_data.to_csv(to_file, sep=',', index=None)
    print("saved to", to_file)

