from os import path as osp
from torch.utils import data as data
from utils.registry import DATASET_REGISTRY
import numpy as np
import cv2
import os
import torch
import glob
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from utils.model_utils import get_model_total_params

@DATASET_REGISTRY.register()
class Vid4(Dataset):
    """
    Vid4 数据集加载类。用于加载典型的超分辨率测试集 Vid4。
    """
    def __init__(self, data_dir, transform, gop_size=1, overlay=True):
        super(Vid4, self).__init__()
     
        self.vid4_dir = data_dir
        self.total_len = len(os.listdir(self.vid4_dir))
        self.scenes = ('calendar', 'city', 'foliage', 'walk')
        self.gop_size = gop_size # 图像组大小
        self.alist = []
       
        for s in self.scenes:
            s_ls = sorted(glob.glob(self.vid4_dir+'/'+s+'/*.png'))
            if self.gop_size == 1:
                seq_len = len(s_ls)
            else:
                seq_len = self.gop_size
            
            # 将视频序列划分为指定长度的图像组
            for i in range(0, len(s_ls), seq_len-1):
                if (i+seq_len) >= len(s_ls):
                    break
                self.alist.append(s_ls[i:i+seq_len])
        self.transform = transform # 图像转换操作（如 ToTensor）

    def load_img(self, image_path):
        """ 加载并转换图像列表。"""
        HR = []
        for each in image_path:
            img_gt = Image.open(each).convert('RGB')
            img_gt = self.transform(img_gt)
            HR.append(img_gt)
        
        return HR

    def __getitem__(self, index):
        """ 获取一个视频片段及其对应的场景名称。"""
        GT = self.load_img(self.alist[index])
        this_scene = [each.split('/')[-2]+'/'+each.split('/')[-1] for each in self.alist[index]]
       
        GT = torch.stack(GT, dim=0) # 堆叠帧，形状为 [T, C, H, W]
        return this_scene, GT.permute(1, 0, 2, 3) # 返回场景信息和 [C, T, H, W] 形状的张量

    def __len__(self):
        """ 返回总的视频片段数量。"""
        return len(self.alist)