import numpy as np, mot_3d.tracklet as tracklet, mot_3d.utils as utils
from scipy.optimize import linear_sum_assignment
from .frame_data import FrameData
from .update_info_data import UpdateInfoData
from .data_protos import BBox, Validity
from .association import associate_dets_to_tracks
from . import visualization


class MOTModel:
    def __init__(self, configs):
        self.trackers = list()         # tracker for each single tracklet
        self.frame_count = 0           # record for the frames
        self.count = 0                 # record the obj number to assign ids
        self.time_stamp = None         # the previous time stamp

        self.configs = configs
        self.match_type = configs['running']['match_type']
        self.score_threshold = configs['running']['score_threshold']
        self.asso = configs['running']['asso']
        self.asso_thres = configs['running']['asso_thres'][self.asso]

        self.max_age = configs['running']['max_age_since_update']
        self.min_hits = configs['running']['min_hits_to_birth']
    
    
    def frame_mot(self, input_data: FrameData):
        """ For each frame input, generate the latest mot results
        Args:
            input_data (FrameData): input data, including detection bboxes and ego information
        Returns:
            tracks on this frame: [(bbox0, id0), (bbox1, id1), ...]
        """
        self.frame_count += 1
        # initialize the time stamp on frame 0
        if self.time_stamp is None:
            self.time_stamp = input_data.time_stamp

        matched, unmatched_dets, unmatched_trks = self.forward_step_trk(input_data)

        time_lag = input_data.time_stamp - self.time_stamp
        # update the matched tracks
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                for k in range(len(matched)):
                    if matched[k][1] == t:
                        d = matched[k][0]
                        break
                aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}
                update_info = UpdateInfoData(mode=1, bbox=input_data.dets[d], ego=input_data.ego, 
                    frame_index=self.frame_count, pc=input_data.pc, 
                    dets=input_data.dets, aux_info=aux_info)
                trk.update(update_info)
            else:
                result_bbox, update_mode, aux_info = trk.get_state(),0,None
                aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}
                update_info = UpdateInfoData(mode=update_mode, bbox=result_bbox, 
                    ego=input_data.ego, frame_index=self.frame_count, 
                    pc=input_data.pc, dets=input_data.dets, aux_info=aux_info)
                trk.update(update_info)
        
        # create new tracks for unmatched detections
        for index in unmatched_dets:
            aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}
            track = tracklet.Tracklet(self.configs, self.count, input_data.dets[index], input_data.det_types[index], 
                self.frame_count, aux_info=aux_info, time_stamp=input_data.time_stamp)
            self.trackers.append(track)
            self.count += 1
        
        # remove dead tracks
        track_num = len(self.trackers)
        for index, trk in enumerate(reversed(self.trackers)):
            if trk.death(self.frame_count):
                self.trackers.pop(track_num - 1 - index)
        
        # output the results
        result = list()
        for trk in self.trackers:
            state_string = trk.state_string(self.frame_count)
            result.append((trk.get_state(), trk.id, state_string, trk.det_type))
        
        # wrap up and update the information about the mot trackers
        self.time_stamp = input_data.time_stamp
        for trk in self.trackers:
            trk.sync_time_stamp(self.time_stamp)

        return result
    
    def forward_step_trk(self, input_data: FrameData):
        dets = input_data.dets
        det_indexes = [i for i, det in enumerate(dets) if det.s >= self.score_threshold]
        dets = [dets[i] for i in det_indexes]

        # prediction and association
        trk_preds = list()
        for trk in self.trackers:
            trk_preds.append(trk.predict(input_data.time_stamp, input_data.aux_info['is_key_frame']))
        
        # for m-distance association
        trk_innovation_matrix = None
        if self.asso == 'm_dis':
            trk_innovation_matrix = [trk.compute_innovation_matrix() for trk in self.trackers] 

        matched, unmatched_dets, unmatched_trks = associate_dets_to_tracks(dets, trk_preds, 
            self.match_type, self.asso, self.asso_thres, trk_innovation_matrix)
        
        for k in range(len(matched)):
            matched[k][0] = det_indexes[matched[k][0]]
        for k in range(len(unmatched_dets)):
            unmatched_dets[k] = det_indexes[unmatched_dets[k]]
        return matched, unmatched_dets, unmatched_trks
    