import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets.ytb_vos import YoutubeVOSDataset
from datasets.ytb_vis import YoutubeVISDataset
from datasets.saliency_modular import SaliencyDataset
from datasets.vipseg import VIPSegDataset
from datasets.mvimagenet import MVImageNetDataset
from datasets.sam import SAMDataset
from datasets.uvo import UVODataset
from datasets.uvo_val import UVOValDataset
from datasets.mose import MoseDataset
from datasets.vitonhd import VitonHDDataset
from datasets.fashiontryon import FashionTryonDataset
from datasets.lvis import LvisDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from torch.utils.data import ConcatDataset
from cldm.hack import disable_verbosity, enable_sliced_attention
from omegaconf import OmegaConf

save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()

# Configs
resume_path = 'path/to/weight'
batch_size = 16
logger_freq = 1000
learning_rate = 1e-5
sd_locked = False
only_mid_control = False
n_gpus = 2
accumulate_grad_batches=1

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./configs/anydoor.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Datasets
DConf = OmegaConf.load('./configs/datasets.yaml')
dataset1 = YoutubeVOSDataset(**DConf.Train.YoutubeVOS)  
dataset2 =  SaliencyDataset(**DConf.Train.Saliency) 
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

image_data = [dataset2, dataset6, dataset12]
video_data = [dataset1, dataset3, dataset4, dataset7, dataset9, dataset10 ]
tryon_data = [dataset8, dataset11]
threed_data = [dataset5]

# The ratio of each dataset is adjusted by setting the __len__ 
dataset = ConcatDataset( image_data + video_data + tryon_data +  threed_data + video_data + tryon_data +  threed_data  )
dataloader = DataLoader(dataset, num_workers=8, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=n_gpus, strategy="ddp", precision=16, accelerator="gpu", callbacks=[logger], progress_bar_refresh_rate=1, accumulate_grad_batches=accumulate_grad_batches)

# Train!
trainer.fit(model, dataloader)
