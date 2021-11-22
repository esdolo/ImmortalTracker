path=$1
echo "Processing Waymo_info, Dataset Path=${path}"
mkdir ./data
mkdir ./data/waymo
python preparedata/waymo/time_stamp.py --data_folder ${path}
python preparedata/waymo/ego_info.py --data_folder ${path} --process 10