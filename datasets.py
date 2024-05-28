import os
import random
import torch
import h5py
import numpy as np
import json
import math
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from helpers.bbox_helper import get_loc_label, get_ctr_label
from helpers.vsumm_helper import get_keyshot_summ

import matplotlib.pyplot as plt

class VideoSummDataset(object):
    def __init__(self, keys, args=None):
        self.keys = keys
        self.video_dict = h5py.File('{}/{}/feature/eccv16_dataset_{}_google_pool5.h5'.format(args.data_root, args.dataset, args.dataset.lower()), 'r')
        
        key = self.keys[0]
        video_name = key.split('/')[-1]
        video_file = self.video_dict[video_name]

        video = torch.from_numpy(video_file['features'][...].astype(np.float32)) # [T, 1024]

        text_feature_path = '{}/{}/feature/text_roberta.npy'.format(args.data_root, args.dataset)
        text_feature_dict = np.load(text_feature_path, allow_pickle=True).item()
        video_id_list = text_feature_dict.keys()

        self.text_dict = {}
        for video_id in video_id_list:
            self.text_dict[video_id] = torch.from_numpy(text_feature_dict[video_id]).to(torch.float32)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        video_name = key.split('/')[-1]
        video_file = self.video_dict[video_name]

        video = torch.from_numpy(video_file['features'][...].astype(np.float32)) # [T, 1024]
        # print("찐비디오사이즈 ", video.size())

        text = self.text_dict[video_name] # [T, 1024]

        gtscore = video_file['gtscore'][...].astype(np.float32) # [T]
        # print("gtsocrer size T ", gtscore.shape)
        change_points = video_file['change_points'][...].astype(np.int32) # [S, 2], S: number of segments, each row stores indices of a segment
        n_frames = video_file['n_frames'][...].astype(np.int32) # [N], N: number of frames, N = T * 15
        n_frame_per_seg = video_file['n_frame_per_seg'][...].astype(np.int32) # [S], indicates number of frames in each segment
        # print("segment size S ", n_frame_per_seg.shape)
        picks = video_file['picks'][...].astype(np.int32) # [T], posotions of subsampled frames in original video

        user_summary = np.zeros(0, dtype=np.float32)
        if 'user_summary' in video_file:
            user_summary = video_file['user_summary'][...].astype(np.float32)

        gtscore -= gtscore.min()
        gtscore /= gtscore.max()

        keyshot_summ, gtscore_upsampled = get_keyshot_summ(gtscore, change_points, n_frames, n_frame_per_seg, picks) #Generate keyshot-based video summary
        target = keyshot_summ[::15]

        # input embedding
        video_cls_label = target
        video_loc_label = get_loc_label(target)
        video_ctr_label = get_ctr_label(target, video_loc_label) #centerness

        video_cls_label = torch.from_numpy(video_cls_label)
        video_loc_label = torch.from_numpy(video_loc_label)
        video_ctr_label = torch.from_numpy(video_ctr_label)

### segment embedding인가
        num_frame = video.shape[0]
        num_sentence = text.shape[0] # 차원 구함
        frame_sentence_ratio = int(math.ceil(num_frame / num_sentence)) # 프레임/문장 비율
        text_cls_label = np.zeros((num_sentence), dtype=bool) #text 차원만큼 zeros 만듦
        for j in range(num_sentence):
            start_frame = j * frame_sentence_ratio
            end_frame = min((j + 1) * frame_sentence_ratio, num_frame)
            if video_cls_label[start_frame: end_frame].any(): # 하나라도 true면 트루 반환
                text_cls_label[j] = True

        # input embedding
        text_loc_label = get_loc_label(text_cls_label)
        text_ctr_label = get_ctr_label(text_cls_label, text_loc_label)

        text_cls_label = torch.from_numpy(text_cls_label)
        text_loc_label = torch.from_numpy(text_loc_label)
        text_ctr_label = torch.from_numpy(text_ctr_label)
        
        # alignment guided self attention mask
        video_to_text_mask = torch.zeros((num_frame, num_sentence), dtype=torch.long)
        text_to_video_mask = torch.zeros((num_sentence, num_frame), dtype=torch.long)
        for j in range(num_sentence):
            start_frame = j * frame_sentence_ratio
            end_frame = min((j + 1) * frame_sentence_ratio, num_frame)
            video_to_text_mask[start_frame: end_frame, j] = 1
            text_to_video_mask[j, start_frame : end_frame] = 1

        mask_video = torch.ones(num_frame, dtype=torch.long)
        mask_text = torch.ones(num_sentence, dtype=torch.long)

        # New video feature generation using the text feature
        text = self.text_dict[video_name]

        ratio = 0.15
        return video, text, mask_video, mask_text, video_cls_label, video_loc_label, video_ctr_label, text_cls_label, text_loc_label, text_ctr_label, \
            user_summary, n_frames, ratio, n_frame_per_seg, picks, change_points, video_to_text_mask, text_to_video_mask


def worker_init_fn(worker_id):
    """
    Re-seed each worker process to preserve reproducibility
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return

def my_collate_fn(batch):
    batched_output_list = []
    for i in range(len(batch[0])):
        batched_output = [item[i] for item in batch]
        batched_output_list.append(batched_output)
    return batched_output_list