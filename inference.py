import cv2
import einops
import numpy as np
import torch
import random
from datasets.data_utils import (
    get_bbox_from_mask,
    expand_image_mask,
    pad_to_square,
    sobel,
    expand_bbox,
    box2squre,
    box_in_box,
)
import albumentations as A

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

save_memory = False


def aug_data_mask(image, mask):
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ]
    )
    transformed = transform(image=image.astype(np.uint8), mask=mask)
    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]
    return transformed_image, transformed_mask


def process_pairs(ref_image, ref_mask, tar_image, tar_mask):
    # ========= Reference ===========
    # ref expand
    ref_box_yyxx = get_bbox_from_mask(ref_mask)

    # ref filter mask
    ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (
        1 - ref_mask_3
    )

    y1, y2, x1, x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2, x1:x2, :]
    ref_mask = ref_mask[y1:y2, x1:x2]

    ratio = np.random.randint(12, 13) / 10
    masked_ref_image, ref_mask = expand_image_mask(
        masked_ref_image, ref_mask, ratio=ratio
    )
    ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)

    # to square and resize
    masked_ref_image = pad_to_square(masked_ref_image, pad_value=255, random=False)
    masked_ref_image = cv2.resize(masked_ref_image, (224, 224)).astype(np.uint8)

    ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value=0, random=False)
    ref_mask_3 = cv2.resize(ref_mask_3, (224, 224)).astype(np.uint8)
    ref_mask = ref_mask_3[:, :, 0]

    # ref aug
    masked_ref_image_aug = masked_ref_image  # aug_data(masked_ref_image)

    # collage aug
    masked_ref_image_compose, ref_mask_compose = (
        masked_ref_image,
        ref_mask,
    )  # aug_data_mask(masked_ref_image, ref_mask)
    masked_ref_image_aug = masked_ref_image_compose.copy()
    ref_mask_3 = np.stack([ref_mask_compose, ref_mask_compose, ref_mask_compose], -1)
    ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose / 255)

    # ========= Target ===========
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1, 1.2])

    # crop
    tar_box_yyxx_crop = expand_bbox(tar_image, tar_box_yyxx, ratio=[1.5, 3])  # 1.2 1.6
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop)  # crop box
    y1, y2, x1, x2 = tar_box_yyxx_crop

    cropped_target_image = tar_image[y1:y2, x1:x2, :]
    tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
    y1, y2, x1, x2 = tar_box_yyxx

    # collage
    ref_image_collage = cv2.resize(ref_image_collage, (x2 - x1, y2 - y1))
    ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2 - x1, y2 - y1))
    ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)

    collage = cropped_target_image.copy()
    collage[y1:y2, x1:x2, :] = ref_image_collage

    collage_mask = cropped_target_image.copy() * 0.0
    collage_mask[y1:y2, x1:x2, :] = 1.0

    # the size before pad
    H1, W1 = collage.shape[0], collage.shape[1]
    cropped_target_image = pad_to_square(
        cropped_target_image, pad_value=0, random=False
    ).astype(np.uint8)
    collage = pad_to_square(collage, pad_value=0, random=False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value=-1, random=False).astype(
        np.uint8
    )

    # the size after pad
    H2, W2 = collage.shape[0], collage.shape[1]
    cropped_target_image = cv2.resize(cropped_target_image, (512, 512)).astype(
        np.float32
    )
    collage = cv2.resize(collage, (512, 512)).astype(np.float32)
    collage_mask = (
        cv2.resize(collage_mask, (512, 512)).astype(np.float32) > 0.5
    ).astype(np.float32)

    masked_ref_image_aug = masked_ref_image_aug / 255
    cropped_target_image = cropped_target_image / 127.5 - 1.0
    collage = collage / 127.5 - 1.0
    collage = np.concatenate([collage, collage_mask[:, :, :1]], -1)

    item = dict(
        ref=masked_ref_image_aug.copy(),
        jpg=cropped_target_image.copy(),
        hint=collage.copy(),
        extra_sizes=np.array([H1, W1, H2, W2]),
        tar_box_yyxx_crop=np.array(tar_box_yyxx_crop),
    )
    return item


def crop_back(pred, tar_image, extra_sizes, tar_box_yyxx_crop):
    H1, W1, H2, W2 = extra_sizes
    y1, y2, x1, x2 = tar_box_yyxx_crop
    pred = cv2.resize(pred, (W2, H2))
    m = 5  # maigin_pixel

    if W1 == H1:
        tar_image[y1 + m : y2 - m, x1 + m : x2 - m, :] = pred[m:-m, m:-m]
        return tar_image

    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:, pad1:-pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1:-pad2, :, :]

    gen_image = tar_image.copy()
    gen_image[y1 + m : y2 - m, x1 + m : x2 - m, :] = pred[m:-m, m:-m]
    return gen_image


def inference_single_image(
    model,
    ddim_sampler,
    ref_image,
    ref_mask,
    tar_image,
    tar_mask,
    guidance_scale=5.0,
):
    item = process_pairs(ref_image, ref_mask, tar_image, tar_mask)
    ref = item["ref"] * 255
    tar = item["jpg"] * 127.5 + 127.5
    hint = item["hint"] * 127.5 + 127.5

    hint_image = hint[:, :, :-1]
    hint_mask = item["hint"][:, :, -1] * 255
    hint_mask = np.stack([hint_mask, hint_mask, hint_mask], -1)
    ref = cv2.resize(ref.astype(np.uint8), (512, 512))

    seed = random.randint(0, 65535)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    ref = item["ref"]
    tar = item["jpg"]
    hint = item["hint"]
    num_samples = 1

    control = torch.from_numpy(hint.copy()).float().cuda()
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, "b h w c -> b c h w").clone()

    clip_input = torch.from_numpy(ref.copy()).float().cuda()
    clip_input = torch.stack([clip_input for _ in range(num_samples)], dim=0)
    clip_input = einops.rearrange(clip_input, "b h w c -> b c h w").clone()

    guess_mode = False
    H, W = 512, 512

    cond = {
        "c_concat": [control],
        "c_crossattn": [model.get_learned_conditioning(clip_input)],
    }
    un_cond = {
        "c_concat": None if guess_mode else [control],
        "c_crossattn": [
            model.get_learned_conditioning(
                [torch.zeros((1, 3, 224, 224))] * num_samples
            )
        ],
    }
    shape = (4, H // 8, W // 8)

    if save_memory:
        model.low_vram_shift(is_diffusing=True)

    # ====
    num_samples = 1  # gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
    image_resolution = 512  # gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
    strength = 1  # gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
    guess_mode = False  # gr.Checkbox(label='Guess Mode', value=False)
    # detect_resolution = 512  #gr.Slider(label="Segmentation Resolution", minimum=128, maximum=1024, value=512, step=1)
    ddim_steps = (
        50  # gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
    )
    scale = guidance_scale  # gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
    seed = (
        -1
    )  # gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
    eta = 0.0  # gr.Number(label="eta (DDIM)", value=0.0)

    model.control_scales = (
        [strength * (0.825 ** float(12 - i)) for i in range(13)]
        if guess_mode
        else ([strength] * 13)
    )  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates = ddim_sampler.sample(
        ddim_steps,
        num_samples,
        shape,
        cond,
        verbose=False,
        eta=eta,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=un_cond,
    )
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (
        (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
        .cpu()
        .numpy()
    )  # .clip(0, 255).astype(np.uint8)

    result = x_samples[0][:, :, ::-1]
    result = np.clip(result, 0, 255)

    pred = x_samples[0]
    pred = np.clip(pred, 0, 255)[1:, :, :]
    sizes = item["extra_sizes"]
    tar_box_yyxx_crop = item["tar_box_yyxx_crop"]
    gen_image = crop_back(pred, tar_image, sizes, tar_box_yyxx_crop)
    return gen_image
