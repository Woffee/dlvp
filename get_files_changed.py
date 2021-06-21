"""
获取每一个 commit 的 file changes.

e.g.

{
"sha": "597d371329d390267e1ad194e7979bc79466cf0b",
"filename": "mm/memory_hotplug.c",
"status": "modified",
"additions": 8,
"deletions": 7,
"changes": 15,
"blob_url": "https://github.com/torvalds/linux/blob/08dff7b7d629807dbb1f398c68dd9cd58dd657a1/mm/memory_hotplug.c",
"raw_url": "https://github.com/torvalds/linux/raw/08dff7b7d629807dbb1f398c68dd9cd58dd657a1/mm/memory_hotplug.c",
"contents_url": "https://api.github.com/repos/torvalds/linux/contents/mm/memory_hotplug.c?ref=08dff7b7d629807dbb1f398c68dd9cd58dd657a1",
"patch": "@@ -512,19 +512,20 @@ int __ref online_pages(unsigned long pfn, unsigned long nr_pages)\n \n \tzone->present_pages += onlined_pages;\n \tzone->zone_pgdat->node_present_pages += onlined_pages;\n-\tif (need_zonelists_rebuild)\n-\t\tbuild_all_zonelists(NULL, zone);\n-\telse\n-\t\tzone_pcp_update(zone);\n+\tif (onlined_pages) {\n+\t\tnode_set_state(zone_to_nid(zone), N_HIGH_MEMORY);\n+\t\tif (need_zonelists_rebuild)\n+\t\t\tbuild_all_zonelists(NULL, zone);\n+\t\telse\n+\t\t\tzone_pcp_update(zone);\n+\t}\n \n \tmutex_unlock(&zonelists_mutex);\n \n \tinit_per_zone_wmark_min();\n \n-\tif (onlined_pages) {\n+\tif (onlined_pages)\n \t\tkswapd_run(zone_to_nid(zone));\n-\t\tnode_set_state(zone_to_nid(zone), N_HIGH_MEMORY);\n-\t}\n \n \tvm_total_pages = nr_free_pagecache_pages();\n "
}

old ones:
drwxr-xr-x  5 ww6 users  129 Jun 18 17:22 FFmpeg/
drwxr-xr-x  4 ww6 users   40 Jun 14 11:04 httpd_/
drwxr-xr-x  5 ww6 users   62 Jun 14 11:04 imap_/
drwxr-xr-x  5 ww6 users   59 Jun 14 11:04 libexpat_/
drwxr-xr-x  5 ww6 users   61 Jun 14 11:03 libsndfile_/
drwxr-xr-x  5 ww6 users 4096 Jun 18 17:53 linux_/
drwxr-xr-x  5 ww6 users   58 Jun 14 11:03 openssl_/
drwxr-xr-x  5 ww6 users   58 Jun 14 11:03 php_/
drwxr-xr-x  4 ww6 users 4096 Jun 14 11:03 qemu_/
drwxr-xr-x  5 ww6 users   58 Jun 14 11:03 radare2_/
drwxr-xr-x  4 ww6 users   40 Jun 14 11:03 suricata_/
drwxr-xr-x  4 ww6 users   40 Jun 14 11:03 tcpdump_/
drwxr-xr-x  5 ww6 users   59 Jun 14 11:03 wildmidi_/


@Time    : 6/20/21
@Author  : Wenbo
"""

import pandas as pd
import json
import os
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import time
import traceback
import sys

def get_response(url):
    try:
        return json.loads(urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0',
               'Authorization': 'token bd560be2f4e2d1ec018bbbc405c26f82d1eed16b',
               'Content-Type':'application/json',
               'Accept':'application/json'})).read())
    except HTTPError as e:
        if e.code == 429:
            time.sleep(10)
            return get_response(url)
        elif e.code == 404:
            print("\n not found:" + url+ "！")
            return ""
        elif e.code == 403:
            time.sleep(600)
            return get_response(url)
        else:
            traceback.print_exc(file=sys.stdout)
            print("reason", e)
            return ""
        raise
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        print("reason", e)
        print("\n skip get_response:"+url+ "！")
        return ""

def get_file_changed(user_proj, commit_id):
    # commit_id = "f833c53cb596e9e1792949f762e0b33661822748"
    # url = "https://api.github.com/repos/libsndfile/libsndfile/commits/" + commit_id

    url = "https://api.github.com/repos/%s/commits/%s" % (user_proj, commit_id)

    response = get_response(url)
    if not (response is None) and not (response == ""):
        # commit_message = response["commit"]["message"]
        # project_before = response["parents"][0]["sha"]
        # project_after = commit_id
        files_changed = ""
        j = 0
        for i in response["files"]:
            if j < len(response["files"]) - 1:
                files_changed = files_changed + json.dumps(i) + "<_**next**_>"
            else:
                files_changed = files_changed + json.dumps(i)
            j += 1
        return files_changed
    return ""


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

commit_log_files = BASE_DIR + "/commit_log_files.txt"

to_file = BASE_DIR + "/data/commit_file_changes.json"

if __name__ == '__main__':
    finished = []
    if os.path.exists(to_file):
        with open(to_file) as f:
            for line in f.readlines():
                l = line.strip()
                if l=="":
                    continue
                item = json.loads(l)
                if item['commit_id'] not in finished:
                    finished.append(item['commit_id'])

    # commit_id = "f833c53cb596e9e1792949f762e0b33661822748"
    # url = "https://api.github.com/repos/libsndfile/libsndfile/commits/" + commit_id
    # response = get_response(url)
    # if not (response is None) and not (response == ""):
    #     commit_message = response["commit"]["message"]
    #     project_before = response["parents"][0]["sha"]
    #     project_after = commit_id
    #     files_changed = ""
    #     j = 0
    #     for i in response["files"]:
    #         if j < len(response["files"]) - 1:
    #             files_changed = files_changed + json.dumps(i) + "<_**next**_>"
    #         else:
    #             files_changed = files_changed + json.dumps(i)
    #         j += 1
    #     print(files_changed)
    # exit()

    projs = {}
    with open(commit_log_files, "r") as f:
        for line in f.readlines():
            l = line.strip()
            if l=="":
                continue
            arr = l.split("/")
            full_name = arr[1]
            if full_name[-1] == "_":
                full_name = arr[1][:-1]
            shot_name = arr[2]

            pos = l.find("commit_log")

            projs[full_name] = {
                'short_name': shot_name,
                'folder_path': l[:pos]
            }

    for k in projs:
        commits_file = BASE_DIR + "/data/%s_commits_in_order.csv" % k
        if not os.path.exists(commits_file):
            print("=== file not existed: %s" % commits_file)
            continue

        print("=== now:", k)

        folder_path = BASE_DIR + "/" + projs[k]['folder_path']
        short_name = projs[k]['short_name']

        df = pd.read_csv(commits_file)

        for index, row in df.iterrows():

            commit_id = row['commit_id']
            if commit_id in finished:
                print(commit_id + " === finished")
                continue
            print(commit_id)

            # get commit message
            cmd = "cd %s && git --no-pager log --pretty=oneline -n 1 %s" % (folder_path, commit_id)
            print(cmd)
            p = os.popen(cmd)
            x = p.read()
            arr = x.strip().split(" ", 1)
            commit_message = ""
            if len(arr) > 1:
                commit_message = arr[1]

            # get username & project_name
            cmd = "cd %s && git remote -v" % folder_path
            print(cmd)
            p = os.popen(cmd)
            x = p.read()

            github_link = ""
            user_proj = ""





            for line in x.split("\n"):
                s = line.find("https")
                e = line.find(" (fetch)")
                if s > -1 and e > -1:
                    github_link = line[s:e]
                    print("github link:", github_link)
                    user_proj = github_link.replace("https://github.com/", "").replace(".git", "")
                    print("user_proj:",user_proj)
                    break

            if user_proj!="":
                files_changed = get_file_changed(user_proj, commit_id)
                with open(to_file, "a") as fw:
                    fw.write(json.dumps({
                        "commit_id": commit_id,
                        "commit_message": commit_message,
                        "full_project_name": k,
                        "short_project_name": short_name,
                        "github_link": github_link,
                        "files_changed": files_changed
                    }) + "\n")
                print("saved to ", to_file)

            # tmp_file = BASE_DIR + "/data/tmp/%s_logs.txt" % commit_id
            # cmd = "cd %s && git --no-pager show %s > %s" % (folder_path, commit_id, tmp_file)
            # print(cmd)
            # p = os.popen(cmd)
            # x = p.read()
            #
            # print("tmp_file:",tmp_file)
            #
            # log_text = ""
            # with open(tmp_file, "r", encoding="utf8", errors='ignore') as f:
            #     log_text = f.read()
            #
            # # print(log_text)
            # if log_text == "":
            #     continue
            #
            # for line in log_text.split("\n"):
            #     l = line.strip()



        break
