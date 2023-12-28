import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
from .data_utils import * 
from .base import BaseDataset
import albumentations as A

class FashionTryonDataset(BaseDataset):
    def __init__(self, image_dir):
        self.image_root = image_dir
        self.data =os.listdir(self.image_root)
        self.size = (512,512)
        self.clip_size = (224,224)
        self.dynamic = 2

    def __len__(self):
        return 5000

    def aug_data(self, image):
        transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            ])
        transformed = transform(image=image.astype(np.uint8))
        transformed_image = transformed["image"]
        return transformed_image

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
        cloth_dir = os.path.join(self.image_root, self.data[idx])
        ref_image_path = os.path.join(cloth_dir, 'target.jpg')

        ref_image = cv2.imread(ref_image_path)
        ref_image = cv2.cvtColor(ref_image.copy(), cv2.COLOR_BGR2RGB)

        ref_mask_path = os.path.join(cloth_dir,'mask.jpg')
        ref_mask = cv2.imread(ref_mask_path)[:,:,0] > 128

        target_dirs = [i for i in os.listdir(cloth_dir ) if '.jpg' not in i]
        target_dir_name = np.random.choice(target_dirs)

        target_image_path = os.path.join(cloth_dir, target_dir_name + '.jpg')
        target_image= cv2.imread(target_image_path)
        tar_image = cv2.cvtColor(target_image.copy(), cv2.COLOR_BGR2RGB)

        target_mask_path = os.path.join(cloth_dir, target_dir_name, 'segment.png')
        tar_mask= cv2.imread(target_mask_path)[:,:,0]
        target_mask =  tar_mask == 7        
        kernel = np.ones((3, 3), dtype=np.uint8)
        tar_mask = cv2.erode(target_mask.astype(np.uint8), kernel, iterations=3)

        item_with_collage = self.process_pairs(ref_image, ref_mask, tar_image, tar_mask, max_ratio = 1.0)
        sampled_time_steps = self.sample_timestep()
        item_with_collage['time_steps'] = sampled_time_steps
        return item_with_collage



 