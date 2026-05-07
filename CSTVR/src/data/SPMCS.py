import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from PIL import Image
import data.core_bicubic as core_bicubic
from torchvision import transforms

class SPMCS_arb(Dataset):
    """
    SPMCS 数据集加载类，支持任意尺度的缩放。
    用于加载训练或测试视频序列。
    """
    def __init__(self, data_dir, scale, t_scale):
        super(SPMCS_arb, self).__init__()
        self.name_list = sorted(os.listdir(data_dir))
        self.seq_list = []
        for name in self.name_list:
            this_dir = os.path.join(data_dir, name, 'HR')
            # 获取每个视频序列中的所有高分辨率图像路径
            self.seq_list.append([os.path.join(this_dir, each) for each in sorted(os.listdir(this_dir))])
        self.scale = scale # 空间缩放因子
        self.t_scale = t_scale # 时间缩放因子
       
   
    def load_img(self, image_list):
        """ 加载图像列表并转换为 numpy 数组列表。"""
        HR = []
        for img in image_list:
            img_gt = Image.open(img).convert('RGB')
            HR.append(img_gt)
        HR = [np.asarray(h) for h in HR]
        
        return HR

    def __getitem__(self, index):
        """
        根据索引获取一个视频序列及其对应的低分辨率版本。
        """
        output_list = []
        GT = self.load_img(self.seq_list[index]) 
        GT = np.asarray(GT).astype('float32')/255.0  # 归一化到 [0, 1]
        t = GT.shape[0] # 帧数
        h = GT.shape[1] # 高度
        w = GT.shape[2] # 宽度
        c = GT.shape[3] # 通道数
   
        # 转换为 PyTorch 张量，形状为 [T, C, H, W]
        GT = torch.from_numpy(GT).permute(0, 3, 1, 2).contiguous()
       
        # 根据时间缩放因子选择低分辨率帧的索引
        LRindex = [int(i) for i in range(GT.shape[0]) if i % self.t_scale == 0]
        for j in LRindex:
            output_list.append(self.seq_list[index][j])

        # 使用双三次插值生成低分辨率图像
        LR = core_bicubic.imresize(GT[LRindex, :, :, :], sizes=(round(h/self.scale), round(w/self.scale)))
        
        crop_shape = (h, w)
        return GT, LR, self.name_list[index], crop_shape, output_list

    def __len__(self):
        """ 返回视频序列的总数。"""
        return len(self.name_list)