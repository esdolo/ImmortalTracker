import os, argparse, json, numpy as np
from pyquaternion import Quaternion
import sys
sys.path.append('.')
from mot_3d.data_protos import BBox, Validity
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='immortal')
parser.add_argument('--obj_types', type=str, default='car,bus,trailer,truck,pedestrian,bicycle,motorcycle')
parser.add_argument('--result_folder', type=str, default='./mot_results/nuscenes/')
parser.add_argument('--data_folder', type=str, default='./data/nuscenes/')
parser.add_argument('--mode', type=str, default='2hz', choices=['20hz', '2hz'])
parser.add_argument('--test', action='store_true', default=False)
args = parser.parse_args()

def bbox_array2nuscenes_format(bbox_array):
    translation = bbox_array[:3].tolist()
    size = bbox_array[4:7].tolist()
    size = [size[1], size[0], size[2]]
    velocity = [0.0, 0.0]
    score = bbox_array[-1]

    yaw = bbox_array[3]
    rot_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0],
                           [np.sin(yaw),  np.cos(yaw), 0, 0],
                           [0,            0,           1, 0],
                           [0,            1,           0, 1]])
    q = Quaternion(matrix=rot_matrix)
    rotation = q.q.tolist()

    sample_result = {
        'translation':    translation,
        'size':           size,
        'velocity':       velocity,
        'rotation':       rotation,
        'tracking_score': score
    }
    return sample_result


def main(name, obj_types, data_folder, result_folder, output_folder):
    for obj_type in obj_types:
        print('CONVERTING {:}'.format(obj_type))
        summary_folder = os.path.join(result_folder, 'summary', obj_type)
        file_names = sorted(os.listdir(os.path.join(data_folder, 'ego_info')))
        token_info_folder = os.path.join(data_folder, 'token_info')
    
        results = dict()
        pbar = tqdm(total=len(file_names))
        for file_index, file_name in enumerate(file_names):
            segment_name = file_name.split('.')[0]
            token_info = json.load(open(os.path.join(token_info_folder, '{:}.json'.format(segment_name)), 'r'))
            mot_results = np.load(os.path.join(summary_folder, '{:}.npz'.format(segment_name)), allow_pickle=True)
    
            ids, bboxes, states, types = \
                mot_results['ids'], mot_results['bboxes'], mot_results['states'], mot_results['types']
            frame_num = len(ids)
            for frame_index in range(frame_num):
                sample_token = token_info[frame_index]
                results[sample_token] = list()
                frame_bboxes, frame_ids, frame_types, frame_states = \
                    bboxes[frame_index], ids[frame_index], types[frame_index], states[frame_index]
                
                frame_obj_num = len(frame_ids)
                for i in range(frame_obj_num):
                    # if not Validity.valid(frame_states[i]):
                    #     continue
                    if not Validity.agein2(frame_states[i]):
                        continue
                    sample_result = bbox_array2nuscenes_format(frame_bboxes[i])
                    sample_result['sample_token'] = sample_token
                    sample_result['tracking_id'] = frame_types[i] + '_' + str(frame_ids[i])
                    sample_result['tracking_name'] = frame_types[i]
                    results[sample_token].append(sample_result)
            pbar.update(1)
        pbar.close()
        submission_file = {
            'meta': {
                'use_camera': False, 'use_lidar': True, 'use_radar': False, 'use_map': False, 'use_external': False
            },
            'results': results
        }
    
        f = open(os.path.join(output_folder, obj_type, 'results.json'), 'w')
        json.dump(submission_file, f)
        f.close()
    return 


if __name__ == '__main__':
    if args.test:
        result_folder = os.path.join(args.result_folder, 'test')
        data_folder = os.path.join(args.data_folder, 'test')
    else:
        result_folder = os.path.join(args.result_folder, 'validation')
        data_folder = os.path.join(args.data_folder, 'validation')

    if args.mode == '2hz':
        result_folder = result_folder + '_2hz'
        data_folder = data_folder + '_2hz'
    elif args.mode == '20hz':
        result_folder = result_folder + '_20hz'
        data_folder = data_folder + '_20hz'

    result_folder = os.path.join(result_folder, args.name)
    obj_types = args.obj_types.split(',')
    output_folder = os.path.join(result_folder, 'results')
    for obj_type in obj_types:
        tmp_output_folder = os.path.join(result_folder, 'results', obj_type)
        if not os.path.exists(tmp_output_folder):
            os.makedirs(tmp_output_folder)
    
    main(args.name, obj_types, data_folder, result_folder, output_folder)