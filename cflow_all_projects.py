"""


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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    finished_file = "data/finished_github.txt"
    finished = []
    if os.path.exists(finished_file):
        with open(finished_file, "r") as f:
            for line in f.read().strip().split("\n"):
                if line.strip() != "":
                    finished.append(line)

    projects_stats_file = "data/projects_statistics.csv"
    df = pd.read_csv(projects_stats_file)
    df = df.fillna("")

    for index, row in df.iterrows():
        github = row['Github']
        proj = row['Project']

        print("=== now: %d, github: %s" % (index, github))

        if github == "" or github in finished:
            continue

        # json_file = save_path + "/" + proj.lower().replace(" ", "_") + "_vul_list.json"
        data_type = proj.lower().replace(" ", "_")
        vul_list_file = BASE_DIR + "/data/vul_list/%s_vul_list.csv" % data_type

        cmd = "python cflow.py --data_type %s --vul_list_file %s --github %s" % (data_type, vul_list_file, github)
        p = os.popen(cmd)
        x = p.read()

        with open(finished_file, "a") as fw:
            fw.write(github + "\n")