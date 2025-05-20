import os
from tqdm import tqdm
import shutil
import subprocess
import re
import skimage.metrics
import numpy as np
import torch
import cv2 as cv
from network.lpips import LPIPS


def crop_image(gt_mask, patch_size, *args):
    """
    :param gt_mask: (H, W)
    :param patch_size: resize the cropped patch to the given patch_size
    :param args: some images with shape of (H, W, C)
    """
    mask_uv = np.argwhere(gt_mask > 0.)
    min_v, min_u = mask_uv.min(0)
    max_v, max_u = mask_uv.max(0)
    pad_size = 50
    min_v = (min_v - pad_size).clip(0, gt_mask.shape[0])
    min_u = (min_u - pad_size).clip(0, gt_mask.shape[1])
    max_v = (max_v + pad_size).clip(0, gt_mask.shape[0])
    max_u = (max_u + pad_size).clip(0, gt_mask.shape[1])
    len_v = max_v - min_v
    len_u = max_u - min_u
    max_size = max(len_v, len_u)

    cropped_images = []
    for image in args:
        if image is None:
            cropped_images.append(None)
        else:
            cropped_image = np.ones((max_size, max_size, 3), dtype=image.dtype)
            if len_v > len_u:
                start_u = (max_size - len_u) // 2
                cropped_image[:, start_u: start_u + len_u] = image[min_v: max_v, min_u: max_u]
            else:
                start_v = (max_size - len_v) // 2
                cropped_image[start_v: start_v + len_v, :] = image[min_v: max_v, min_u: max_u]

            cropped_image = cv.resize(cropped_image, (patch_size, patch_size), interpolation=cv.INTER_LINEAR)
            cropped_images.append(cropped_image)

    if len(cropped_images) > 1:
        return cropped_images
    else:
        return cropped_images[0]


class LPIPSMetric:
    def __init__(self, device="cuda"):
        self.lpips_net = LPIPS(net='vgg').to(device)

    @staticmethod
    def to_tensor(array, device='cuda'):
        if isinstance(array, np.ndarray):
            array = torch.from_numpy(array).to(device)
        elif isinstance(array, torch.Tensor):
            array = array.to(device)
        else:
            raise TypeError('Invalid type of array.')
        return array

    @staticmethod
    def cut_rect(img):
        h, w = img.shape[:2]
        size = max(h, w)
        img_ = torch.ones((size, size, img.shape[2])).to(img)
        if h < w:
            img_[:h] = img
        else:
            img_[:, :w] = img
        return img_

    def compute_lpips(self, src, tar, device='cuda'):
        src = self.to_tensor(src, device)
        tar = self.to_tensor(tar, device)
        if src.shape[0] != src.shape[1]:
            src = self.cut_rect(src)
            tar = self.cut_rect(tar)
        lpips = self.lpips_net.forward(src.permute(2, 0, 1)[None], tar.permute(2, 0, 1)[None], normalize=True).mean()
        return lpips.item()


def compute_psnr(src, tar):
    psnr = skimage.metrics.peak_signal_noise_ratio(tar, src, data_range=1)
    return psnr


def compute_ssim(src, tar):
    ssim = skimage.metrics.structural_similarity(src, tar, multichannel=True, data_range=1, channel_axis=-1)
    return ssim


@torch.inference_mode()
def compute_metrics(ours_dir, gt_dir, mask_dir, frame_list, device="cuda", patch_size=512):
    ours_metrics = {"psnr": 0, "ssim": 0, "lpips": 0, "fid": 0}

    os.makedirs('./tmp_quant/', exist_ok=True)
    shutil.rmtree('./tmp_quant')
    os.makedirs('./tmp_quant/ours', exist_ok=True)
    os.makedirs('./tmp_quant/gt', exist_ok=True)

    lpips_computer = LPIPSMetric(device)
    count = 0
    for frame_id in tqdm(frame_list, desc="Computing metrics..."):
        ours_img = (cv.imread(ours_dir + '/%08d.jpg' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        gt_img = (cv.imread(gt_dir + '/%08d.jpg' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        mask_img = cv.imread(mask_dir + '/%08d.jpg' % frame_id, cv.IMREAD_UNCHANGED) > 128
        gt_img[~mask_img] = 1.

        ours_img_cropped, gt_img_cropped = crop_image(
            mask_img,
            patch_size,
            ours_img,
            gt_img
        )

        cv.imwrite('./tmp_quant/ours/%08d.png' % frame_id, (ours_img_cropped * 255).astype(np.uint8))
        cv.imwrite('./tmp_quant/gt/%08d.png' % frame_id, (gt_img_cropped * 255).astype(np.uint8))

        if ours_img is not None:
            ours_metrics["psnr"] += compute_psnr(ours_img, gt_img).item()
            ours_metrics["ssim"] += compute_ssim(ours_img, gt_img).item()
            ours_metrics["lpips"] += lpips_computer.compute_lpips(ours_img_cropped, gt_img_cropped)
            count += 1

    ours_metrics["psnr"] /= count
    ours_metrics["ssim"] /= count
    ours_metrics["lpips"] /= count

    print('Computing FID...')
    output = subprocess.check_output(
        'python -m pytorch_fid --device cuda {} {}'.format('./tmp_quant/ours', './tmp_quant/gt'), shell=True)
    output = output.decode('utf-8')
    s = float(re.findall(r"FID:\D+(\d+.?\d+)", output)[0])
    ours_metrics["fid"] = s
    return ours_metrics
