#now_time=`date +"%Y-%m-%d-%H-%M"`
#
#data_type="linux"
#log_file="../log/${data_type}_${now_time}.log"
#
#echo python cflow.py --log_file ${log_file} --data_type ${data_type}
#python cflow.py --log_file ${log_file} --data_type ${data_type}


python function2vec.py --cve_jsons_path data/jsons/FFmpeg_jsons \
--all_functions_file data/function2vec/all_functions.csv \
--all_func_trees_file data/function2vec/all_functions_with_trees.csv \
--all_func_trees_json_file data/function2vec/all_functions_with_trees.json \
--all_func_embedding_file data/function2vec/all_func_embedding_file.csv