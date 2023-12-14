
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


class ht_bisenet():
    def __init__(self):

        # uncomment the following line if you want to reduce cpu usage, see issue #231
        #  torch.set_num_threads(4)

        torch.set_grad_enabled(False)
        np.random.seed(123)

        self.weight_path = '/home/ht/ht_code/seg/BiSeNet/model_final_v2_city.pth'
        self.config = '/media/ht/新加卷/ht_code/ht_elevation_map_research/BiSeNet/configs/bisenetv2_city.py'

        cfg = set_cfg_from_file(self.config)

        self.palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

        # define model
        self.net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='eval')
        self.net.load_state_dict(torch.load(self.weight_path, map_location='cpu'), strict=False)
        self.net.eval()
        self.net.cuda()

        # prepare data
        self.to_tensor = T.ToTensor(
            mean=(0.3257, 0.3690, 0.3223),  # city, rgb
            std=(0.2112, 0.2148, 0.2115),
        )

    def __call__(
            self,
            im,
    ) :
        im = im[:, :, ::-1]
        im = self.to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()

        # shape divisor
        org_size = im.size()[2:]
        new_size = [math.ceil(el / 32) * 32 for el in im.size()[2:]]
        # print(new_size)
        # for i in range(50):
        #     t1 = time.time()
            # inference
        im = F.interpolate(im, size=new_size, align_corners=False, mode='bilinear')
        out = self.net(im)[0]
        out = F.interpolate(out, size=org_size, align_corners=False, mode='bilinear')
        out = out.argmax(dim=1)

            # t2 = time.time()

            # print(t2 - t1)

        # visualize
        out = out.squeeze().detach().cpu().numpy()
        """使用 self.palette 对输出张量 out 进行颜色映射。out 中的每个值作为索引，
        从 self.palette 中选择对应的颜色。这样，pred 变量将成为一个具有相同形状的张量，
        其中的每个元素都是一个 RGB 颜色值，对应于模型输出的每个类别"""
        pred = self.palette[out]
        # print(out.shape)
        out = out.astype(np.uint8)
        # cv2.imwrite('./res.jpg', pred)
        # pred 是语义分割的可视化结果, out 是图像每个像素分割出来对应类别
        return pred,out
# 判断输入层在elemap 还是 plug

if __name__ == "__main__":
    ht_bisenet_predictor = ht_bisenet()
    img_path = '/media/ht/新加卷/ht_code/ht_elevation_map_research/BiSeNet/example.png'
    im = cv2.imread(img_path)
    pred,out = ht_bisenet_predictor(im)
    cv2.imshow('0',pred)
    cv2.imwrite('0.jpg', pred)
    cv2.waitKey(0)