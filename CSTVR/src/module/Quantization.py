import torch
import torch.nn as nn
import torch.nn.functional as F

class Quant(torch.autograd.Function):
    """
    量化函数，将输入限制在 [0, 1] 范围内，并量化为 256 个级别（8位）。
    在前向传播中执行量化，在反向传播中直接传递梯度（Straight-Through Estimator, STE）。
    """

    @staticmethod
    def forward(ctx, input):
        input = torch.clamp(input, 0, 1)
        output = (input * 255.).round() / 255.
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Quantization(nn.Module):
    """量化层包装器。"""
    def __init__(self):
        super(Quantization, self).__init__()

    def forward(self, input):
        return Quant.apply(input)
    
class Quantize_ste(nn.Module):
    """
    带有直通估计器（STE）的量化层。
    手动实现裁剪和量化，同时保持梯度流。
    """
    def __init__(self, min_val, max_val):
        super(Quantize_ste, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        # 使用 ReLU 实现可微分裁剪
        x_clipped_min = x + F.relu(self.min_val - x)
        x_clipped = x_clipped_min - F.relu(x_clipped_min - self.max_val)
        # 量化并使用 STE 传递梯度
        return (torch.round(x_clipped*255.)/255. - x_clipped).detach() + x_clipped

class DifferentiableClipping(torch.autograd.Function):
    """
    可微分裁剪函数。
    在前向传播中执行裁剪和量化，在反向传播中根据输入值调整梯度。
    """
    @staticmethod
    def forward(ctx, in_ten, min_val, max_val):
        # 保存用于反向传播的张量
        ctx.save_for_backward(in_ten)
        ctx.min_val = min_val
        ctx.max_val = max_val
        # 裁剪到 [min_val, max_val]
        x_clipped_min = in_ten + F.relu(min_val - in_ten)
        x_clipped = x_clipped_min - F.relu(x_clipped_min - max_val)

        # 量化到 8 位
        x_clipped = (x_clipped * 255.).round() / 255.
        return x_clipped

    @staticmethod
    def backward(ctx, grad_output):
        in_ten, = ctx.saved_tensors
        min_val = ctx.min_val
        max_val = ctx.max_val
        grad_input = grad_output.clone()
        # 如果输入超出范围，则放大梯度（此处设置为 2 是一种实验性的做法）
        grad_input[in_ten <= min_val] = 2
        grad_input[in_ten >= max_val] = 2
        # 返回输入的梯度，min_val 和 max_val 不需要梯度
        return grad_input, None, None

class DifferentiableQuantization(nn.Module):
    """可微分量化层包装器。"""
    def __init__(self, min_val=0.0, max_val=1.0):
        super(DifferentiableQuantization, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, input):
        return DifferentiableClipping.apply(input, self.min_val, self.max_val)


# 示例用法
if __name__ == "__main__":
    quantization_layer = DifferentiableQuantization(0.0, 1.0)
    input_tensor = torch.randn(10, requires_grad=True)
    output = quantization_layer(input_tensor)
    output.backward(torch.ones_like(input_tensor))  # 反向传播测试梯度计算
