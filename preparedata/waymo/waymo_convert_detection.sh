file_path=$1
detection_name=$2
python preparedata/waymo/detection.py --file_name ${file_path} --name ${detection_name}