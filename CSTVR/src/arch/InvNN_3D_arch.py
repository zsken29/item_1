import torch.nn as nn
from module.inv_module import InvBlockExp, ST_Pixelshuffle_Downsampling

class my_InvNN(nn.Module):
	"""
	自定义的 3D 可逆神经网络架构。
	由多个下采样操作和可逆块交替堆叠而成。
	"""
	def __init__(self, mode='test', channel_in=3, channel_out=3, subnet_constructor=None, block_num=2, down_num=2):
		super(my_InvNN, self).__init__()
		self.mode = mode
		operations = []

		current_channel = channel_in
		for i in range(down_num):
			# 添加空时下采样操作
			b = ST_Pixelshuffle_Downsampling()
			operations.append(b)
			current_channel *= 4 # 下采样后通道数增加
			for j in range(block_num):
				# 添加可逆块
				channel_out = 3
				spilt_mode = 0
				b = InvBlockExp(subnet_constructor, current_channel, channel_out, clamp=1, spilt_mode=spilt_mode)
				operations.append(b)
		self.operations = nn.ModuleList(operations)

	def forward(self, x, rev=False, cal_jacobian=False):
		"""
		前向或逆向传播。
		参数:
			rev (bool): 是否为逆向传播。
			cal_jacobian (bool): 是否计算雅可比行列式的对数。
		"""
		out = x
		jacobian = 0
		if not rev:
			# 正向传播：按顺序执行操作
			for op in self.operations:
				out = op.forward(out, rev)
				if cal_jacobian:
					jacobian += op.jacobian(out, rev)
		else:
			# 逆向传播：按相反顺序执行操作
			for op in reversed(self.operations):
				out = op.forward(out, rev)
				if cal_jacobian:
					jacobian += op.jacobian(out, rev)
		if cal_jacobian:
			return out, jacobian
		else:
			return out

