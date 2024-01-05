# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
from cog import BasePredictor, Input, Path
import os
import cv2
import time
import torch
import einops
import random
import subprocess
import numpy as np
from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_state_dict
from cldm.hack import disable_verbosity
from datasets.data_utils import * 
from omegaconf import OmegaConf


save_memory = False
MODEL_URL = "https://weights.replicate.delivery/default/ali-vilab/anydoor.tar"
MODEL_CACHE="checkpoints"


def download(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

def process_pairs(ref_image, ref_mask, tar_image, tar_mask, max_ratio = 0.8, enable_shape_control = False):
    # ========= Reference ===========
    # ref expand 
    ref_box_yyxx = get_bbox_from_mask(ref_mask)

    # ref filter mask 
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3)

    y1,y2,x1,x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]
    ref_mask = ref_mask[y1:y2,x1:x2]

    ratio = np.random.randint(11, 15) / 10 #11,13
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)

    # to square and resize
    masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False)
    masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), (224,224) ).astype(np.uint8)

    ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value = 0, random = False)
    ref_mask_3 = cv2.resize(ref_mask_3.astype(np.uint8), (224,224) ).astype(np.uint8)
    ref_mask = ref_mask_3[:,:,0]

    # collage aug 
    masked_ref_image_compose, ref_mask_compose =  masked_ref_image, ref_mask
    ref_mask_3 = np.stack([ref_mask_compose,ref_mask_compose,ref_mask_compose],-1)
    ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose/255)

    # ========= Target ===========
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1,1.2]) #1.1  1.3
    tar_box_yyxx_full = tar_box_yyxx
    
    # crop
    tar_box_yyxx_crop =  expand_bbox(tar_image, tar_box_yyxx, ratio=[1.3, 3.0])   
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
    y1,y2,x1,x2 = tar_box_yyxx_crop

    cropped_target_image = tar_image[y1:y2,x1:x2,:]
    cropped_tar_mask = tar_mask[y1:y2,x1:x2]

    tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
    y1,y2,x1,x2 = tar_box_yyxx

    # collage
    ref_image_collage = cv2.resize(ref_image_collage.astype(np.uint8), (x2-x1, y2-y1))
    ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
    ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)

    collage = cropped_target_image.copy() 
    collage[y1:y2,x1:x2,:] = ref_image_collage

    collage_mask = cropped_target_image.copy() * 0.0
    collage_mask[y1:y2,x1:x2,:] = 1.0
    if enable_shape_control:
        collage_mask = np.stack([cropped_tar_mask,cropped_tar_mask,cropped_tar_mask],-1)

    # the size before pad
    H1, W1 = collage.shape[0], collage.shape[1]

    cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
    collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value = 2, random = False).astype(np.uint8)

    # the size after pad
    H2, W2 = collage.shape[0], collage.shape[1]

    cropped_target_image = cv2.resize(cropped_target_image.astype(np.uint8), (512,512)).astype(np.float32)
    collage = cv2.resize(collage.astype(np.uint8), (512,512)).astype(np.float32)
    collage_mask  = cv2.resize(collage_mask.astype(np.uint8), (512,512),  interpolation = cv2.INTER_NEAREST).astype(np.float32)
    collage_mask[collage_mask == 2] = -1

    masked_ref_image = masked_ref_image  / 255 
    cropped_target_image = cropped_target_image / 127.5 - 1.0
    collage = collage / 127.5 - 1.0 
    collage = np.concatenate([collage, collage_mask[:,:,:1]  ] , -1)
    
    item = dict(ref=masked_ref_image.copy(), jpg=cropped_target_image.copy(), hint=collage.copy(), 
                extra_sizes=np.array([H1, W1, H2, W2]), 
                tar_box_yyxx_crop=np.array( tar_box_yyxx_crop ),
                tar_box_yyxx=np.array(tar_box_yyxx_full),
                 ) 
    return item

def crop_back( pred, tar_image,  extra_sizes, tar_box_yyxx_crop):
    H1, W1, H2, W2 = extra_sizes
    y1,y2,x1,x2 = tar_box_yyxx_crop    
    pred = cv2.resize(pred, (W2, H2))
    m = 5 # maigin_pixel

    if W1 == H1:
        tar_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
        return tar_image

    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:,pad1: -pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1: -pad2, :, :]

    gen_image = tar_image.copy()
    gen_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
    return gen_image

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # if checkpoints folder does not exist, create it
        if not os.path.exists(MODEL_CACHE):
            download(MODEL_URL, MODEL_CACHE)
        
        disable_verbosity()
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        config = OmegaConf.load('./configs/inference.yaml')
        model_ckpt =  config.pretrained_model
        model_config = config.config_file
        model = create_model(model_config).cpu()
        model.load_state_dict(load_state_dict(model_ckpt, location='cuda'))
        self.model = model.cuda()
        self.ddim_sampler = DDIMSampler(model)


    def inference_single_image(self, ref_image, ref_mask, tar_image, tar_mask, strength, ddim_steps, guidance_scale, seed, enable_shape_control):
        item = process_pairs(ref_image, ref_mask, tar_image, tar_mask, enable_shape_control)
        ref = item['ref'] * 255
        tar = item['jpg'] * 127.5 + 127.5
        hint = item['hint'] * 127.5 + 127.5

        hint_image = hint[:,:,:-1]
        hint_mask = item['hint'][:,:,-1] * 255
        hint_mask = np.stack([hint_mask,hint_mask,hint_mask],-1)
        ref = cv2.resize(ref.astype(np.uint8), (512,512))

        seed = random.randint(0, 65535)
        if save_memory:
            self.model.low_vram_shift(is_diffusing=False)

        ref = item['ref']
        tar = item['jpg'] 
        hint = item['hint']
        num_samples = 1
        control = torch.from_numpy(hint.copy()).float().cuda() 
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        clip_input = torch.from_numpy(ref.copy()).float().cuda() 
        clip_input = torch.stack([clip_input for _ in range(num_samples)], dim=0)
        clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w').clone()
        guess_mode = False
        H,W = 512,512
        cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning( clip_input )]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([torch.zeros((1,3,224,224))] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if save_memory:
            self.model.low_vram_shift(is_diffusing=True)

        # ====
        num_samples = 1 #gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
        image_resolution = 512  #gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
        #strength = 1  #gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
        guess_mode = False #gr.Checkbox(label='Guess Mode', value=False)
        #detect_resolution = 512  #gr.Slider(label="Segmentation Resolution", minimum=128, maximum=1024, value=512, step=1)
        #ddim_steps = 50 #gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
        scale = guidance_scale  #gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
        #seed = -1  #gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
        eta = 0.0 #gr.Number(label="eta (DDIM)", value=0.0)

        self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        samples, intermediates = self.ddim_sampler.sample(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond)
        if save_memory:
            self.model.low_vram_shift(is_diffusing=False)

        x_samples = self.model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()

        result = x_samples[0][:,:,::-1]
        result = np.clip(result,0,255)

        pred = x_samples[0]
        pred = np.clip(pred,0,255)[1:,:,:]
        sizes = item['extra_sizes']
        tar_box_yyxx_crop = item['tar_box_yyxx_crop'] 
        gen_image = crop_back(pred, tar_image, sizes, tar_box_yyxx_crop) 
        return gen_image

    def predict(
        self,
        reference_image_path: Path = Input(description="Source Image"),
        reference_image_mask: Path = Input(description="Source Image"),
        bg_image_path: Path = Input(description="Target Image"),
        bg_mask_path: Path = Input(description="Target Image mask"),
        control_strength: float = Input(description="Control Strength", default=1.0, ge=0.0, le=2.0),
        steps: int = Input(description="Steps", default=50, ge=1, le=100),
        guidance_scale: float = Input(description="Guidance Scale", default=4.5, ge=0.1, le=30.0),
        enable_shape_control: bool = Input(description="Enable Shape Control", default=False),
        seed: int = Input(description="Random seed. Leave blank to randomize the seed", default=None),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"Using seed: {seed}")

        save_path = "/tmp/output.png"
        image = cv2.imread(str(reference_image_path), cv2.IMREAD_UNCHANGED)
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        ref_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ref_mask = (cv2.imread(str(reference_image_mask))[:,:,-1] > 128).astype(np.uint8)

        # background image
        back_image = cv2.imread(str(bg_image_path)).astype(np.uint8)
        back_image = cv2.cvtColor(back_image, cv2.COLOR_BGR2RGB)

       # background mask 
        tar_mask = cv2.imread(str(bg_mask_path))[:,:,0] > 128
        tar_mask = tar_mask.astype(np.uint8)
        
        gen_image = self.inference_single_image(
            ref_image,ref_mask, back_image.copy(), tar_mask,
            control_strength, steps, guidance_scale, seed, enable_shape_control)
        h,w = back_image.shape[0], back_image.shape[0]
        ref_image = cv2.resize(ref_image, (w,h))
        vis_image = cv2.hconcat([gen_image])
        cv2.imwrite(save_path, vis_image [:,:,::-1])

        return Path(save_path)