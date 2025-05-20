import json
import os

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import torch
import torch.utils.data
import numpy as np
import cv2 as cv
from tqdm import tqdm
import config
from utils.net_util import to_cuda
from dataset.dataset_mv_rgb import MvRgbDatasetAvatarReX
from argparse import ArgumentParser
from importlib.machinery import SourceFileLoader
from comparison_body_only_avatars import compute_metrics


def parse_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--model_name', type=str, default='example')
    arg_parser.add_argument('--avatar', type=str, default='zzr')
    arg_parser.add_argument('--camera', type=int, default='7', help="13 or 7")
    args = arg_parser.parse_args()
    return args


@torch.inference_mode()
def main():
    torch.manual_seed(31359)
    np.random.seed(31359)

    args = parse_args()

    load_model = SourceFileLoader(
        "load_model", os.path.abspath(os.path.join("model", args.model_name, 'load_model.py'))
    ).load_module()
    avatar_net = load_model.load_model(args.avatar, "cuda")

    config.opt.update({"test": {"data": {"frame_range": [0, 500]}}})
    if args.avatar == "zzr":
        config.opt["test"]["data"]["data_dir"] = os.path.abspath(os.path.join("data", "avatarrex", "avatarrex_zzr"))
    elif args.avatar == "lbn1":
        config.opt["test"]["data"]["data_dir"] = os.path.abspath(os.path.join("data", "avatarrex", "avatarrex_lbn1"))
    elif args.avatar == "lbn2":
        config.opt["test"]["data"]["data_dir"] = os.path.abspath(os.path.join("data", "avatarrex", "avatarrex_lbn2"))
    else:
        raise ValueError
    opt = config.opt

    dataset = MvRgbDatasetAvatarReX(**opt['test']['data'], training=False)

    output_dir = os.path.abspath(os.path.join('test_results', args.model_name, f"{args.avatar}_camera{args.camera}"))
    os.makedirs(output_dir, exist_ok=True)

    time_start = torch.cuda.Event(enable_timing=True)
    time_end = torch.cuda.Event(enable_timing=True)

    all_time = 0
    for idx in tqdm(range(len(dataset)), desc='Rendering avatars...'):

        img_scale = 1.0
        cam_id = args.camera
        intr = dataset.intr_mats[cam_id].copy()
        intr[:2] *= img_scale
        extr = dataset.extr_mats[cam_id].copy()
        img_h, img_w = int(dataset.img_heights[cam_id] * img_scale), int(
            dataset.img_widths[cam_id] * img_scale)

        item = dataset.getitem(
            idx,
            training=False,
            extr=extr,
            intr=intr,
            img_w=img_w,
            img_h=img_h
        )
        items = to_cuda(item, add_batch=False)

        if 'smpl_pos_map' not in items:
            avatar_net.get_pose_map(items)

        # Render
        torch.cuda.synchronize()
        time_start.record()

        output = avatar_net.render(items, bg_color=(1., 1., 1.), use_pca=False)

        time_end.record()
        torch.cuda.synchronize()

        # print('Rendering avatar costs %.4f secs' % (time_start.elapsed_time(time_end) / 1000.))
        all_time += time_start.elapsed_time(time_end)

        # Saving images
        rgb_map = output['rgb_map']
        rgb_map.clip_(0., 1.)
        rgb_map = (rgb_map * 255).to(torch.uint8).cpu().numpy()
        cv.imwrite(output_dir + '/%08d.jpg' % item['data_idx'], rgb_map)

        torch.cuda.empty_cache()

    cam_name = dataset.cam_names[args.camera]
    gt_dir = os.path.join(dataset.data_dir, cam_name)
    mask_dir = os.path.join(gt_dir, 'mask', 'pha')
    metrics = compute_metrics(ours_dir=output_dir,
                              gt_dir=gt_dir,
                              mask_dir=mask_dir,
                              frame_list=list(range(500)),
                              device="cuda", patch_size=512)
    metrics["mean_inference_time"] = all_time / (1000. * len(dataset))

    with open(f'./{args.model_name}_{args.avatar}_{args.camera}.json', 'w') as f:
        json.dump(metrics, f, indent='  ')


if __name__ == '__main__':
    main()
