now_time=`date +"%Y-%m-%d-%H-%M"`

data_type="linux"
log_file="../log/${data_type}_${now_time}.log"

echo python cflow.py --log_file ${log_file} --data_type ${data_type}
python cflow.py --log_file ${log_file} --data_type ${data_type}
