path=$1
echo "Processing nuScenes_info, Dataset Path=${path}"
mkdir ./data
mkdir ./data/nuscenes
python preparedata/nuscenes/time_stamp.py --raw_data_folder ${path}
python preparedata/nuscenes/ego_pose.py --raw_data_folder ${path}
python preparedata/nuscenes/token_info.py --raw_data_folder ${path}