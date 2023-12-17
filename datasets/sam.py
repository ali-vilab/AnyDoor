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

class SAMDataset(BaseDataset):
    def __init__(self, sub1, sub2, sub3, sub4):
        image_mask_dict = {}
        self.data = []
        self.register_subset(sub1)
        self.register_subset(sub2)
        self.register_subset(sub3)
        self.register_subset(sub4)
        self.size = (512,512)
        self.clip_size = (224,224)
        self.dynamic = 0

    def register_subset(self, path):
        data = os.listdir(path)
        data = [ os.path.join(path, i) for i in data if '.json' in i]
        self.data = self.data + data

    def get_sample(self, idx):
        # ==== get pairs =====
        json_path = self.data[idx]
        image_path = json_path.replace('.json', '.jpg')

        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
        annotation = data['annotations']

        valid_ids = []
        for i in range(len(annotation)):
            area = annotation[i]['area']
            if area > 100 * 100 * 5:
                valid_ids.append(i)

        chosen_id = np.random.choice(valid_ids)
        mask = mask_utils.decode(annotation[chosen_id]["segmentation"] )
        # ======================

        image = cv2.imread(image_path)
        ref_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        tar_image = ref_image
        
        ref_mask = mask
        tar_mask = mask
        item_with_collage = self.process_pairs(ref_image, ref_mask, tar_image, tar_mask)
        sampled_time_steps = self.sample_timestep()
        item_with_collage['time_steps'] = sampled_time_steps
        return item_with_collage

    def __len__(self):
        return 20000

    def check_region_size(self, image, yyxx, ratio, mode = 'max'):
        pass_flag = True
        H,W = image.shape[0], image.shape[1]
        H,W = H * ratio, W * ratio
        y1,y2,x1,x2 = yyxx
        h,w = y2-y1,x2-x1
        if mode == 'max':
            if h > H or w > W:
                pass_flag = False
        elif mode == 'min':
            if h < H or w < W:
                pass_flag = False
        return pass_flag



        
