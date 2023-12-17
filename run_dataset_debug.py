from datasets.ytb_vos import YoutubeVOSDataset
from datasets.ytb_vis import YoutubeVISDataset
from datasets.saliency_modular import SaliencyDataset
from datasets.vipseg import VIPSegDataset
from datasets.mvimagenet import MVImageNetDataset
from datasets.sam import SAMDataset
from datasets.dreambooth import DreamBoothDataset
from datasets.uvo import UVODataset
from datasets.uvo_val import UVOValDataset
from datasets.mose import MoseDataset
from datasets.vitonhd import VitonHDDataset
from datasets.fashiontryon import FashionTryonDataset
from datasets.lvis import LvisDataset
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
import numpy as np 
import cv2
from omegaconf import OmegaConf

# Datasets
DConf = OmegaConf.load('./configs/datasets.yaml')
dataset1 = YoutubeVOSDataset(**DConf.Train.YoutubeVOS)  
dataset2 = SaliencyDataset(**DConf.Train.Saliency) 
dataset3 = VIPSegDataset(**DConf.Train.VIPSeg) 
dataset4 = YoutubeVISDataset(**DConf.Train.YoutubeVIS) 
dataset5 = MVImageNetDataset(**DConf.Train.MVImageNet)
dataset6 = SAMDataset(**DConf.Train.SAM)
dataset7 = UVODataset(**DConf.Train.UVO.train)
dataset8 = VitonHDDataset(**DConf.Train.VitonHD)
dataset9 = UVOValDataset(**DConf.Train.UVO.val)
dataset10 = MoseDataset(**DConf.Train.Mose)
dataset11 = FashionTryonDataset(**DConf.Train.FashionTryon)
dataset12 = LvisDataset(**DConf.Train.Lvis)

dataset = dataset5


def vis_sample(item):
    ref = item['ref']* 255
    tar = item['jpg'] * 127.5 + 127.5
    hint = item['hint'] * 127.5 + 127.5
    step = item['time_steps']
    print(ref.shape, tar.shape, hint.shape, step.shape)

    ref = ref[0].numpy()
    tar = tar[0].numpy()
    hint_image = hint[0, :,:,:-1].numpy()
    hint_mask = hint[0, :,:,-1].numpy()
    hint_mask = np.stack([hint_mask,hint_mask,hint_mask],-1)
    ref = cv2.resize(ref.astype(np.uint8), (512,512))
    vis = cv2.hconcat([ref.astype(np.float32), hint_image.astype(np.float32), hint_mask.astype(np.float32), tar.astype(np.float32) ])
    cv2.imwrite('sample_vis.jpg',vis[:,:,::-1])


dataloader = DataLoader(dataset, num_workers=8, batch_size=4, shuffle=True)
print('len dataloader: ', len(dataloader))
for data in dataloader:  
    vis_sample(data) 


