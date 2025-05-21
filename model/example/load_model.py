import os
import config
from network.avatar import AvatarNet
import torch

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'


def load_model(avatar, device):
    path = os.path.dirname(os.path.abspath(__file__))

    if avatar == "zzr":
        data_path = os.path.join(path, '../../data/avatarrex/avatarrex_zzr')
        model_path = os.path.join(path, './net_zzr.pt')
    elif avatar == "lbn1":
        data_path = os.path.join(path, '../../data/avatarrex/avatarrex_lbn1')
        model_path = os.path.join(path, './net_lbn1.pt')
    elif avatar == "lbn2":
        data_path = os.path.join(path, '../../data/avatarrex/avatarrex_lbn2')
        model_path = os.path.join(path, './net_lbn2.pt')
    else:
        raise ValueError

    config.opt.update({"train": {"data": {"data_dir": data_path}}})
    config.device = torch.device(device)
    avatar_net = AvatarNet({"with_viewdirs": True, "random_style": False}).to(config.device)
    net_dict = torch.load(model_path)
    avatar_net.load_state_dict(net_dict['avatar_net'])

    for p in avatar_net.parameters():
        p.requires_grad = False
    avatar_net.eval()
    return avatar_net


if __name__ == '__main__':
    net = load_model("zzr", "cuda")
    print(net)
