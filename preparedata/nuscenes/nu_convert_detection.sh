file_path=$1
detection_name=$2
python preparedata/nuscenes/detection.py --file_path ${file_path} --det_name ${detection_name}