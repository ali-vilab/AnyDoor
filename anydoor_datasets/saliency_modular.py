import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
from .data_utils import * 
from .base import BaseDataset

class SaliencyDataset(BaseDataset):
    def __init__(self, MSRA_root, TR_root, TE_root, HFlickr_root):
        image_mask_dict = {}

        # ====== MSRA-10k ======
        file_lst = os.listdir(MSRA_root)
        image_lst = [MSRA_root+i for i in file_lst if '.jpg' in i]
        for i in image_lst:
            mask_path = i.replace('.jpg','.png')
            image_mask_dict[i] = mask_path

        # ===== DUT-TR ========
        file_lst = os.listdir(TR_root)
        image_lst = [TR_root+i for i in file_lst if '.jpg' in i]
        for i in image_lst:
            mask_path = i.replace('.jpg','.png').replace('DUTS-TR-Image','DUTS-TR-Mask')
            image_mask_dict[i] = mask_path
        
        # ===== DUT-TE ========
        file_lst = os.listdir(TE_root)
        image_lst = [TE_root+i for i in file_lst if '.jpg' in i]
        for i in image_lst:
            mask_path = i.replace('.jpg','.png').replace('DUTS-TE-Image','DUTS-TE-Mask')
            image_mask_dict[i] = mask_path 
        
        # ===== HFlickr =======
        file_lst = os.listdir(HFlickr_root)
        mask_list = [HFlickr_root+i for i in file_lst if '.png' in i]
        for i in file_lst:
            image_name  = i.split('_')[0] +'.jpg'
            image_path = HFlickr_root.replace('masks', 'real_images') + image_name
            mask_path = HFlickr_root + i
            image_mask_dict[image_path] = mask_path 

        self.image_mask_dict = image_mask_dict
        self.data = list(self.image_mask_dict.keys() )
        self.size = (512,512)
        self.clip_size = (224,224)
        self.dynamic = 0

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

    def get_sample(self, idx):

        # ==== get pairs =====
        image_path = self.data[idx]
        mask_path = self.image_mask_dict[image_path]

        instances_mask = cv2.imread(mask_path)
        if len(instances_mask.shape) == 3:
            instances_mask = instances_mask[:,:,0]
        instances_mask = (instances_mask > 128).astype(np.uint8)
        # ======================
        ref_image = cv2.imread(image_path)
        ref_image = cv2.cvtColor(ref_image.copy(), cv2.COLOR_BGR2RGB)
        tar_image = ref_image

        ref_mask = instances_mask
        tar_mask = instances_mask 
        item_with_collage = self.process_pairs(ref_image, ref_mask, tar_image, tar_mask)
        sampled_time_steps = self.sample_timestep()
        item_with_collage['time_steps'] = sampled_time_steps
        return item_with_collage


        
