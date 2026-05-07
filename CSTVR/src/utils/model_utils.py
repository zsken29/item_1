import torch.nn as nn
import numpy as np
import torch.nn.init as init

def get_model_total_params(model):
    """计算模型的可训练参数总量（以百万为单位）。"""
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return (1.0*params/(1000*1000))


def initialize_weights(net_l, scale=1):
    """使用 Kaiming 初始化方法初始化网络权重。

    参数:
        net_l (nn.Module | list[nn.Module]): 要初始化的网络或网络列表。
        scale (float): 缩放因子，常用于残差块。默认值: 1。
    """
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # 用于残差块
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    """使用 Xavier 初始化方法初始化网络权重。

    参数:
        net_l (nn.Module | list[nn.Module]): 要初始化的网络或网络列表。
        scale (float): 缩放因子。默认值: 1。
    """
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # 用于残差块
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


