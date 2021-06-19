"""
把 wenbo 的数据按照 jiahao 的数据格式，合并到一起。

@Time    : 6/18/21
@Author  : Wenbo
"""
import pandas as pd
import json

jh_data_file = "/Users/woffee/www/vulnerability/MSR_20_Code_vulnerability_CSV_Dataset/all_c_cpp_release2.0.csv"
wb_data_file = "/Users/woffee/www/vulnerability/dlvp/compare_caller_callee/wenbo_data_with_commit_message2.csv"

cves_info_file = "/Users/woffee/www/vulnerability/craw_cve/jiahaodata/cves_info.json"

to_file = "all_c_cpp_release2.1.csv"

columns = ['authentication_required', 'availability_impact', 'cve_id', 'cve_page', 'cwe_id', 'access_complexity',
           'confidentiality_impact', 'integrity_impact', 'publish_date', 'score', 'summary', 'update_date',
           'vulnerability_classification', 'ref_link', 'commit_id', 'commit_message', 'files_changed', 'lang',
           'project', 'version_after_fix', 'version_before_fix']

def read_jh_data(data_file):
    data = {}
    proj_lang = {}
    cve_data = {}

    df = pd.read_csv(data_file)
    df = df.fillna('')

    # 注：同一个 commit 有可能出现在多个 CVE 中。
    # print(len(df))
    # print(len(df['commit_id'].unique()))


    for index, row in df.iterrows():
        cve_id = row['cve_id']
        commit_id = row['commit_id']

        if commit_id == "":
            continue

        p = row['project']
        lang = row['lang']

        if p not in proj_lang.keys():
            proj_lang[p] = lang

        hash_k = cve_id + "_" + commit_id

        if hash_k not in data.keys():
            item = {}
            for c in columns:
                item[c] = row[c]
            data[hash_k] = item
        else:
            print("existed: %s" % hash_k)

        if cve_id not in cve_data.keys():
            item = {}
            for c in columns:
                item[c] = row[c]
            cve_data[cve_id] = item

    return data, cve_data, proj_lang

def read_cves_info(cves_info_file):
    cves_data = {}
    with open(cves_info_file, "r") as f:
        for line in f.readlines():
            l = line.strip()
            if l=="":
                continue
            item = json.loads(l)
            cves_data[ item['cve_id'] ] = item
    return cves_data

if __name__ == '__main__':
    jh_data, jh_cve_data, proj_lang = read_jh_data(jh_data_file)
    cves_data = read_cves_info(cves_info_file)

    df = pd.read_csv(wb_data_file)
    df['cc_sum'] = df['callers_total_before'] + df['callers_total_after'] + df['callees_total_before'] + df[
        'callees_total_after']
    df_wenbo_data = df[ df['cc_sum'] > 0]

    for index, row in df_wenbo_data.iterrows():
        cve_id = row['cve_id']
        commit_id = row['commit_id']
        project = row['project']

        if commit_id == "":
            continue

        hash_k = cve_id + "_" + commit_id
        if hash_k in jh_data.keys():
            continue

        lang = ""
        if project in proj_lang.keys():
            lang = proj_lang[project]

        if cve_id in jh_cve_data.keys():
            info = jh_cve_data[ cve_id ]
            info['ref_link'] = ''
            info['commit_id'] = row['commit_id']
            info['commit_message'] = row['commit_message']
            info['version_after_fix'] = row['commit_id']
            info['version_before_fix'] = row['commit_id'] + "^1"
            info['files_changed'] = ''
            jh_data[hash_k] = info

        elif cve_id in cves_data.keys():
            info = cves_data[cve_id]
            for ii, c_id in enumerate(info['commit_ids']) :
                if c_id == commit_id:
                    item = {
                        'authentication_required': info['Authentication'],
                        'availability_impact': info['Availability Impact'],
                        'cve_id':info['cve_id'],
                        'cve_page':info['cve_page'],
                        'cwe_id':info['CWE ID'],

                        'access_complexity':info['Access Complexity'],
                        'confidentiality_impact':info['Authentication'],
                        'integrity_impact':info['Integrity Impact'],
                        'publish_date':info['publish_date'],
                        'score':info['CVSS Score'],

                        'summary':info['summary'],
                        'update_date':info['update_date'],
                        'vulnerability_classification':info['Vulnerability Type(s)'],
                        'ref_link':info['ref_links'][ii],
                        'commit_id':info['commit_ids'][ii],

                        'commit_message': row['commit_message'],
                        'files_changed': "",
                        'lang': lang,
                        'project': row['project'],
                        'version_after_fix': row['commit_id'],
                        'version_before_fix': row['commit_id'] + "^1"
                    }
                    jh_data[hash_k] = item
        else:
            continue

    jsons_data = []
    for k in jh_data.keys():
        v = jh_data[k]
        jsons_data.append(v)

    to_df = pd.DataFrame(jsons_data)
    # df = pd.json_normalize(jsons_data)
    to_df.to_csv(to_file, index=False)
    print("to_file: %s" % to_file)

