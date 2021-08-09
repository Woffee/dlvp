#now_time=`date +"%Y-%m-%d-%H-%M"`

# Cflow
#data_type="linux"
#log_file="../log/${data_type}_${now_time}.log"
#echo python cflow.py --log_file ${log_file} --data_type ${data_type}
#python cflow.py --log_file ${log_file} --data_type ${data_type}


# Functions to vec
#python function2vec2.py --cur_p 9 \
#--tasks_file cflow/tasks.json \
#--all_func_trees_file data/function2vec2/all_functions_with_trees.csv

# Run GCN
python gcn3.py --tasks_file /data/function2vec3/tasks.json \
  --functions_path /data/function2vec3/functions_jy \
  --embedding_path /data/function2vec4 \
  --model_save_path /data/gcn_models_p3 \
  --learning_rate 0.0001 \
  --epoch 200 \
  --input_dim 128 \
  --hidden_dim 128 \
  --batch_size 16 \
  --lp_path_num 20 \
  --lp_length 60 \
  --lp_dim 128 \
  --lp_w2v_path /data/function2vec4/models/w2v_lp_combine.bin \
  --ns_length 2000 \
  --ns_dim 128 \
  --ns_w2v_path /data/function2vec4/models/w2v_ns.bin