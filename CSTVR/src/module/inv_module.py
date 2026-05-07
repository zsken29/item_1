import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.model_utils as mutil

class HaarDownsampling(nn.Module):
	"""Haar 小波下采样模块。用于可逆神经网络中的下采样操作。"""
	def __init__(self, channel_in):
		super(HaarDownsampling, self).__init__()
		self.channel_in = channel_in

		self.haar_weights = torch.ones(4, 1, 2, 2)

		# H (水平)
		self.haar_weights[1, 0, 0, 1] = -1
		self.haar_weights[1, 0, 1, 1] = -1

		# V (垂直)
		self.haar_weights[2, 0, 1, 0] = -1
		self.haar_weights[2, 0, 1, 1] = -1

		# D (对角)
		self.haar_weights[3, 0, 1, 0] = -1
		self.haar_weights[3, 0, 0, 1] = -1

		self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
		self.haar_weights = nn.Parameter(self.haar_weights)
		self.haar_weights.requires_grad = False

	def forward(self, x, rev=False):
		if not rev:
			# 正向：下采样
			self.elements = x.shape[1] * x.shape[2] * x.shape[3]
			self.last_jac = self.elements / 4 * np.log(1 / 16.)
			
			out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
			out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
			out = torch.transpose(out, 1, 2)
			out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
			return out
		else:
			# 反向：上采样
			self.elements = x.shape[1] * x.shape[2] * x.shape[3]
			self.last_jac = self.elements / 4 * np.log(16.)

			out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
			out = torch.transpose(out, 1, 2)
			out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
			return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups=self.channel_in)

	def jacobian(self, x, rev=False):
		return self.last_jac
		
class Pixelshuffle_Downsampling(nn.Module):
	"""使用 PixelUnshuffle 思想的下采样模块。"""
	def __init__(self):
		super(Pixelshuffle_Downsampling, self).__init__()

	def forward(self, x, rev=False):
		if not rev:
			self.elements = x.shape[1] * x.shape[2] * x.shape[3]
			self.last_jac = self.elements / 4 * np.log(1 / 16.)
			[B, C, H, W] = list(x.size())
			x = x.reshape(B, C, H//2, 2, W//2, 2)
			x = x.permute(0, 1, 3, 5, 2, 4)
			x = x.reshape(B, C*4, H//2, W//2)
			return x
		else:
			self.elements = x.shape[1] * x.shape[2] * x.shape[3]
			self.last_jac = self.elements / 4 * np.log(16.)
			[B, C, H, W] = list(x.size())
			x = x.reshape(B, C//4, 2, 2, H, W)
			x = x.permute(0, 1, 4, 2, 5, 3)
			x = x.reshape(B, C//4, H*2, W*2)
			return x
	def jacobian(self, x, rev=False):
		return self.last_jac

class ST_Pixelshuffle_Downsampling(nn.Module):
	"""空时像素重组下采样模块。"""
	def __init__(self):
		super(ST_Pixelshuffle_Downsampling, self).__init__()
		self.r = 1 # 时间缩放
		self.s = 2 # 空间缩放
	def forward(self, x, rev=False):
		if not rev:
			b, c, t, h, w = x.size()
			self.elements =  c*t*h*w
			self.last_jac = self.elements / 4 * np.log(1 / 16.)
			c_out = c * (self.r * self.s * self.s)
			x = x.view(b, c, t // self.r, self.r, h // self.s, self.s, w // self.s, self.s)
			x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
			x = x.view(b, c_out, t // self.r, h // self.s, w // self.s)
			return x
		else:
			b, c, t, h, w = x.size()
			self.elements =  c*t*h*w
			self.last_jac = self.elements / 4 * np.log(1 / 16.)
			c_out = int( c / (self.r * self.s * self.s))
			x = x.view(b, c_out, self.r, self.s, self.s, t, h, w)
			x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
			x = x.view(b, c_out, self.r * t, self.s * h, self.s * w)
			return x


	def jacobian(self, x, rev=False):
		return self.last_jac


class InvBlockExp(nn.Module):
	"""
	可逆块（Invertible Block），基于指数耦合层设计。
	包含三个子网络：F, G, H。
	"""
	def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.,spilt_mode = 0):
		super(InvBlockExp, self).__init__()

		self.split_len1 = channel_split_num
		self.split_len2 = channel_num - channel_split_num
		self.spilt_mode = spilt_mode
		self.clamp = clamp
		self.F = subnet_constructor(self.split_len2, self.split_len1)
		self.G = subnet_constructor(self.split_len1, self.split_len2)
		self.H = subnet_constructor(self.split_len1, self.split_len2)
	
	def forward(self, x,rev=False):
		if self.spilt_mode==1:
			x = torch.flip(x,dims=[1])

		x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

		
		if not rev:
			y1 = x1 + self.F(x2)
			self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
			y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
		else:
			self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
			y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
			y1 = x1 - self.F(y2)

		return torch.cat((y1, y2), 1)

	def jacobian(self, x, rev=False):
		if not rev:
			jac = torch.sum(self.s)
		else:
			jac = -torch.sum(self.s)
		

		return jac / x.shape[0]




class D2DTInput(nn.Module):
    """
    一个特定的 3D 卷积模块，使用密集连接结构。
    """
    def __init__(self, channel_in, channel_out, init='xavier',\
         gc=32, bias=True,INN_init = True,is_res = False):
        super(D2DTInput, self).__init__()
        self.conv1 = nn.Conv3d(channel_in, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv2 = nn.Conv3d(channel_in + gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv3 = nn.Conv3d(channel_in + 2 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv4 = nn.Conv3d(channel_in + 3 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv5 = nn.Conv3d(channel_in + 4 * gc, channel_out, (3,1,1), 1, (1,0,0), bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if INN_init:
            if init == 'xavier':
                mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            else:
                mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            mutil.initialize_weights(self.conv5, 0)

    def forward(self, x,io_type="3d"):
        
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
   
        return x5
class HaarTransform(nn.Module):
    """Haar 变换模块。"""
    def __init__(self, channel_in):
        super(HaarTransform, self).__init__()
        # 初始化 Haar 权重
        haar_weights = torch.ones(4, 1, 2, 2)
        haar_weights[1, 0, 0, 1] = -1
        haar_weights[1, 0, 1, 1] = -1
        haar_weights[2, 0, 1, 0] = -1
        haar_weights[2, 0, 1, 1] = -1
        haar_weights[3, 0, 1, 0] = -1
        haar_weights[3, 0, 0, 1] = -1
        haar_weights = haar_weights.repeat(channel_in, 1, 1, 1)
        
        # 注册为 buffer，不参与训练
        self.register_buffer('haar_weights', haar_weights)

    def forward(self, x):
        channel_in = x.shape[1]
        out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=channel_in) / 4.0
        out = out.view(x.shape[0], channel_in, 4, x.shape[2] // 2, x.shape[3] // 2)
        out = out.permute(0, 2, 1, 3, 4).contiguous()
        out = out.view(x.shape[0], channel_in * 4, x.shape[2] // 2, x.shape[3] // 2)
        return out


def Haartrans(x):
    """Haar 变换的函数式实现。"""
    channel_in = x.shape[1]
    haar_weights = torch.ones(4, 1, 2, 2).to(x.device)
    haar_weights[1, 0, 0, 1] = -1
    haar_weights[1, 0, 1, 1] = -1
    haar_weights[2, 0, 1, 0] = -1
    haar_weights[2, 0, 1, 1] = -1
    haar_weights[3, 0, 1, 0] = -1
    haar_weights[3, 0, 0, 1] = -1
    haar_weights = torch.cat([haar_weights] * channel_in, 0)
    haar_weights.requires_grad = False
    out = F.conv2d(x, haar_weights, bias=None, stride=2, groups=channel_in) / 4.0
    out = out.reshape([x.shape[0], channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
    out = torch.transpose(out, 1, 2)
    out = out.reshape([x.shape[0], channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
    return out

class DenseBlock(nn.Module):
    """密集连接块（Dense Block）。"""
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5

def Haartrans_back(x):
    channel_in = x.shape[1]//4
    haar_weights = torch.ones(4, 1, 2, 2).to(x.device)
    haar_weights[1, 0, 0, 1] = -1
    haar_weights[1, 0, 1, 1] = -1
    haar_weights[2, 0, 1, 0] = -1
    haar_weights[2, 0, 1, 1] = -1
    haar_weights[3, 0, 1, 0] = -1
    haar_weights[3, 0, 0, 1] = -1
    haar_weights = torch.cat([haar_weights] * channel_in, 0)
    haar_weights.requires_grad = False
    out = x.reshape([x.shape[0], 4, channel_in, x.shape[2], x.shape[3]])
    out = torch.transpose(out, 1, 2)
    out = out.reshape([x.shape[0], channel_in * 4, x.shape[2], x.shape[3]])
    return F.conv_transpose2d(out, haar_weights, bias=None, stride=2, groups=channel_in)

class D2DTInput(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier',\
         shortcut = True,gc=32, bias=True,INN_init = True,is_res = False):
        super(D2DTInput, self).__init__()
        self.conv1 = nn.Conv3d(channel_in, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv2 = nn.Conv3d(channel_in + gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv3 = nn.Conv3d(channel_in + 2 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv4 = nn.Conv3d(channel_in + 3 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv5 = nn.Conv3d(channel_in + 4 * gc, channel_out, (3,1,1), 1, (1,0,0), bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.shortcut = shortcut
        if INN_init:
            if init == 'xavier':
                mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            else:
                mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            mutil.initialize_weights(self.conv5, 0)

    def forward(self, x,io_type="3d"):
        
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        if  self.shortcut:
            x5 = x+x5
        return x5


class D2DTInput_dense(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier',\
         gc=32, bias=True,INN_init = True,is_res = False):
        super(D2DTInput_dense, self).__init__()
        self.conv1 = nn.Conv3d(channel_in, gc, (3,3,3), 1, (1,1,1), bias=bias)
        self.conv2 = nn.Conv3d(channel_in + gc, gc, (3,3,3), 1, (1,1,1), bias=bias)
        self.conv3 = nn.Conv3d(channel_in + 2 * gc, gc, (3,3,3), 1, (1,1,1), bias=bias)
        self.conv4 = nn.Conv3d(channel_in + 3 * gc, gc, (3,3,3), 1, (1,1,1), bias=bias)
        self.conv5 = nn.Conv3d(channel_in + 4 * gc, channel_out, (1,1,1), 1, (0,0,0), bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if INN_init:
            if init == 'xavier':
                mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            else:
                mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            mutil.initialize_weights(self.conv5, 0)

    def forward(self, x,io_type="3d"):
        
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
   
        return x5