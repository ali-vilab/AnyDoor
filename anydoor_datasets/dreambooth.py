import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
from .data_utils import * 
from .base import BaseDataset

class DreamBoothDataset(BaseDataset):
    def __init__(self, fg_dir, bg_dir):
        self.bg_dir = bg_dir
        bg_data = os.listdir(self.bg_dir)
        self.bg_data = [i for i in bg_data if 'mask' in i]
        self.image_dir = fg_dir
        self.data  = os.listdir(self.image_dir)
        self.size = (512,512)
        self.clip_size = (224,224)
        '''
         Dynamic:
            0: Static View, High Quality
            1: Multi-view, Low Quality
            2: Multi-view, High Quality
        '''
        self.dynamic = 1 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = np.random.randint(0, len(self.data)-1)
        item = self.get_sample(idx)
        return item

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

    def get_alpha_mask(self, mask_path):
        image = cv2.imread( mask_path, cv2.IMREAD_UNCHANGED)
        mask = (image[:,:,-1] > 128).astype(np.uint8)
        return mask
        
    def get_sample(self, idx):
        dir_name = self.data[idx]
        dir_path = os.path.join(self.image_dir, dir_name)
        images = os.listdir(dir_path)
        image_name = [i for i in images if '.png' in i][0]
        image_path = os.path.join(dir_path, image_name)

        image = cv2.imread( image_path, cv2.IMREAD_UNCHANGED)
        mask = (image[:,:,-1] > 128).astype(np.uint8)
        image = image[:,:,:-1]

        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        ref_image = image 
        ref_mask = mask
        ref_image, ref_mask = expand_image_mask(image, mask, ratio=1.4)
        bg_idx =  np.random.randint(0, len(self.bg_data)-1)
        
        tar_mask_name = self.bg_data[bg_idx]
        tar_mask_path = os.path.join(self.bg_dir, tar_mask_name)
        tar_image_path = tar_mask_path.replace('_mask','_GT')

        tar_image = cv2.imread(tar_image_path).astype(np.uint8)
        tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)
        tar_mask = (cv2.imread(tar_mask_path) > 128).astype(np.uint8)[:,:,0] 

        item_with_collage = self.process_pairs(ref_image, ref_mask, tar_image, tar_mask)
        sampled_time_steps = self.sample_timestep()
        item_with_collage['time_steps'] = sampled_time_steps
        return item_with_collage

