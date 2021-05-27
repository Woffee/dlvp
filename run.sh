now_time=`date +"%Y-%m-%d-%H-%M"`
log_file="../ffmpeg_${now_time}.log"

data_type="linux"
echo python cflow.py --log_file ${log_file} --data_type ${data_type}
python cflow.py --log_file ${log_file} --data_type ${data_type}
