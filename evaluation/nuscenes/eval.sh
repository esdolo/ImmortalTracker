name=$1
path2nuscenes=$2
echo "./mot_results/nuscenes/${name}/results/results.json"
python nuscenes-devkit/python-sdk/nuscenes/eval/tracking/evaluate.py \
    "./mot_results/nuscenes/${name}/results/results.json" \
    --output_dir "./mot_results/nuscenes/${name}/results/nu_results/" \
    --eval_set "val" \
    --dataroot path2nuscenes
    