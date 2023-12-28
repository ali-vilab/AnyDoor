import cv2 
import json 
import os
from pycocotools import mask as mask_utils
import numpy as np
from tqdm import tqdm

json_path = 'path/UVO/UVO_sparse_train_video_with_interpolation.json'
output_path = "path/UVO/UVO_sparse_train_video_with_interpolation_reorg.json"

with open(json_path, 'r') as fcc_file:
    data = json.load(fcc_file)

info = data['info']
videos = data['videos']
print(len(videos))


uvo_dict = {}
for video in tqdm(videos):
    vid = video['id']
    file_names = video['file_names']
    uvo_dict[vid] = file_names


with open(output_path,"w") as f:
    json.dump(uvo_dict,f)
    print('finish')

