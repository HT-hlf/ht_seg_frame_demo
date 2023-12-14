
import sys
# sys.path.insert(0, '.')
import os

sys.path.insert(0, '/home/ht/ht_code/seg/BiSeNet')
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2

import lib.data.transform_cv2 as T
from lib.models import model_factory
from configs import set_cfg_from_file

import time

# uncomment the following line if you want to reduce cpu usage, see issue #231
#  torch.set_num_threads(4)

torch.set_grad_enabled(False)
np.random.seed(123)

# python tools/demo_video.py --config configs/bisenetv2_coco.py --weight-path /home/ht/ht_code/seg/BiSeNet/model_final_v2_city.pth --input ./video.mp4 --output res.mp4
# args
parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str, default='/media/ht/新加卷/ht_code/ht_elevation_map_research/BiSeNet/configs/bisenetv2_city.py',)
parse.add_argument('--weight-path', type=str, default='/home/ht/ht_code/seg/BiSeNet/model_final_v2_city.pth',)
parse.add_argument('--img-path', dest='img_path', type=str, default='/media/ht/新加卷/ht_code/ht_elevation_map_research/BiSeNet/example.png',)
args = parse.parse_args()
cfg = set_cfg_from_file(args.config)


palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

# define model
net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='eval')
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=False)
net.eval()
net.cuda()

# prepare data
to_tensor = T.ToTensor(
    mean=(0.3257, 0.3690, 0.3223), # city, rgb
    std=(0.2112, 0.2148, 0.2115),
)
im = cv2.imread(args.img_path)[:, :, ::-1]
im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()

# shape divisor
org_size = im.size()[2:]
new_size = [math.ceil(el / 32) * 32 for el in im.size()[2:]]
print(new_size)
for i in range(50):
    t1 = time.time()
    # inference
    im = F.interpolate(im, size=new_size, align_corners=False, mode='bilinear')
    out = net(im)[0]
    out = F.interpolate(out, size=org_size, align_corners=False, mode='bilinear')
    out = out.argmax(dim=1)

    t2 = time.time()

    print(t2-t1)

# visualize
out = out.squeeze().detach().cpu().numpy()
pred = palette[out]
cv2.imwrite('./res.jpg', pred)
