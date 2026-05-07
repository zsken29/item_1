from os import path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.utils.registry import DATASET_REGISTRY

import data.core_bicubic as core_bicubic
import torch
import numpy as np
import random
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image

@DATASET_REGISTRY.register()
class Vimeo_SepTuplet(Dataset):
    """
    Vimeo-90K 数据集加载类。用于加载 7 帧视频序列。
    """
    def __init__(self, opt):
        super(Vimeo_SepTuplet, self).__init__()

        self.opt = opt
        self.load_mode = opt['load_mode'] # 'train' 或 'test'
        self.gt_size = opt['gt_size'] # 高分辨率图像裁剪大小
        # 从文件列表加载图像序列名称
        alist = [line.rstrip() for line in open(os.path.join(opt['image_dir'], opt['file_list']))]
        self.image_filenames = [os.path.join(opt['image_dir'], x) for x in alist]

        self.data_augmentation = opt['data_augmentation'] # 是否进行数据增强
        self.lr_idx = opt['T_index'] # 低分辨率图像对应的索引
        self.down_size = opt['S_scale'] # 空间下采样倍数

    def train_process(self, GT, flip_h=True, rot=True, flip_v=True, converse=True):
        """ 训练数据增强处理。"""
        if random.random() < 0.5 and flip_v:
            GT = [LR[::-1, :, :].copy() for LR in GT]
        if random.random() < 0.5 and flip_h:
            GT = [LR[:, ::-1, :].copy() for LR in GT]
        if rot and random.random() < 0.5:
            GT = [LR.transpose(1, 0, 2).copy() for LR in GT]
        if converse and random.random() < 0.5:
            GT = GT.copy()[::-1]
        return GT

    def load_img(self, image_path):
        """ 加载并裁剪图像序列。"""
        def crop_256(img_list, H, W):
            crop_size = self.gt_size
            rnd_h = random.randint(0, H - crop_size)
            rnd_w = random.randint(0, W - crop_size)

            img_gt_crop = [v[rnd_h:rnd_h + crop_size, rnd_w:rnd_w + crop_size, :] for v in img_list]
            return img_gt_crop

        HR = []
        for img_num in range(7):
            img_gt = Image.open(os.path.join(image_path, 'im{}.png'.format(img_num + 1))).convert('RGB')
            HR.append(img_gt)
        HR = [np.asarray(h) for h in HR]
        H, W = HR[0].shape[:2]
        if self.load_mode == 'train':
            HR = crop_256(HR, H, W)
    
        return HR

    def __getitem__(self, index):
        """ 获取高分辨率序列及其对应的低分辨率版本。"""
        GT = self.load_img(self.image_filenames[index])
        name = self.image_filenames[index]
        if self.load_mode == 'train':
            if self.data_augmentation:
                GT = self.train_process(GT)

        GT = np.asarray(GT).astype('float32')/255.0 # 归一化

        t = GT.shape[0]
        h = GT.shape[1]
        w = GT.shape[2]
        c = GT.shape[3]

        # 转换为 [C, T, H, W] 形状的张量
        GT = torch.from_numpy(GT).permute(3, 0, 1, 2).contiguous()
        
        # 提取指定的低分辨率帧
        LR = GT[:, self.lr_idx, :, :]
        # 如果需要下采样
        if self.down_size != 1:
            LR = core_bicubic.imresize(LR.permute(1, 0, 2, 3).contiguous(), sizes=(h//self.down_size, w//self.down_size))
            LR = LR.permute(1, 0, 2, 3).contiguous()
        return {"imgs": GT, "LR": LR, "path": name}
      
    def __len__(self):
        """ 返回数据集样本总数。"""
        return len(self.image_filenames)