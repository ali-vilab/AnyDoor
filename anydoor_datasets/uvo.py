import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
from .data_utils import * 
from .base import BaseDataset
from pycocotools import mask as mask_utils

class UVODataset(BaseDataset):
    def __init__(self, image_dir, video_json, image_json):
        json_path = video_json 
        with open(json_path, 'r') as fcc_file:
            data = json.load(fcc_file)

        image_json_path = image_json
        with open(image_json_path , 'r') as image_file:
            video_dict = json.load(image_file)

        self.image_root =  image_dir
        self.data = data['annotations']
        self.video_dict = video_dict
        self.size = (512,512)
        self.clip_size = (224,224)
        self.dynamic = 1

    def __len__(self):
        return 25000

    def check_region_size(self, image, yyxx, ratio, mode = 'max'):
        pass_flag = True
        H,W = image.shape[0], image.shape[1]
        H,W = H * ratio, W * ratio
        y1,y2,x1,x2 = yyxx
        h,w = y2-y1,x2-x1
        if mode == 'max':
            if h > H and w > W:
                pass_flag = False
        elif mode == 'min':
            if h < H and w < W:
                pass_flag = False
        return pass_flag

    def get_sample(self, idx):
        ins_anno = self.data[idx]
        video_id = str(ins_anno['video_id'])
        video_names = self.video_dict[video_id]
        masks = ins_anno['segmentations']
        frames = video_names

        # Sampling frames
        min_interval = len(frames)  // 10
        start_frame_index = np.random.randint(low=0, high=len(frames) - min_interval)
        end_frame_index = start_frame_index + np.random.randint(min_interval,  len(frames) - start_frame_index )
        end_frame_index = min(end_frame_index, len(frames) - 1)

        # Get image path
        ref_image_name = frames[start_frame_index]
        tar_image_name = frames[end_frame_index]
        ref_image_path = os.path.join(self.image_root, ref_image_name) 
        tar_image_path = os.path.join(self.image_root, tar_image_name) 

        # Read Image and Mask
        ref_image = cv2.imread(ref_image_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

        tar_image = cv2.imread(tar_image_path)
        tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)

        ref_mask = mask_utils.decode(masks[start_frame_index])
        tar_mask = mask_utils.decode(masks[end_frame_index])

        item_with_collage = self.process_pairs(ref_image, ref_mask, tar_image, tar_mask)
        sampled_time_steps = self.sample_timestep()
        item_with_collage['time_steps'] = sampled_time_steps
        return item_with_collage

