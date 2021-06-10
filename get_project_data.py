import pandas as pd

df = pd.read_csv('train_10425_val_3459_test_3452_im_test_9_15780_im_test_16_26826.csv')

print(df.columns)

"""
Index(['Unnamed: 0', 'Unnamed: 0.1', 'Access Gained', 'Attack Origin',
       'Authentication Required', 'Availability', 'CVE ID', 'CVE Page',
       'CWE ID', 'Complexity', 'Confidentiality', 'Integrity',
       'Known Exploits', 'Publish Date', 'Score', 'Summary', 'Summmary',
       'Update Date', 'Vulnerability Classification', 'add_lines', 'bug',
       'callee', 'callee_num', 'caller', 'caller_num', 'code', 'codeCommit',
       'codeLink', 'codeParent', 'code_callee', 'code_caller', 'commit_id',
       'commit_message', 'del_lines', 'edge_list', 'file_name',
       'files_changed', 'flaw_loc', 'func_after', 'func_before',
       'func_end_num', 'func_loc_num', 'id', 'lang', 'lines_after',
       'lines_before', 'long_path', 'long_path_combine',
       'long_path_combine_context', 'long_path_greedy',
       'long_path_greedy_context', 'longest_path_token_num', 'nodes',
       'parentID', 'patch', 'path_num', 'project', 'project_after',
       'project_before', 'trees', 'trees_callee', 'trees_caller', 'vul',
       'vul_func_with_fix', 'caller_long_path_combine',
       'caller_long_path_greedy', 'caller_long_path', 'caller_nodes',
       'caller_edge_list', 'caller_longest_path_token_num', 'caller_path_num',
       'callee_long_path_combine', 'callee_long_path_greedy',
       'callee_long_path', 'callee_nodes', 'callee_edge_list',
       'callee_longest_path_token_num', 'callee_path_num',
       'caller_long_path_combine_context', 'callee_long_path_combine_context'],
      dtype='object')
"""

df2 = df[df['project']=='Chrome']

df3 = df2[['project','CVE ID','codeLink']]
df3.to_csv("Chrome.csv", sep=',')

