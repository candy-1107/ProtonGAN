import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out



#自定义的线性层，对权重和偏置进行特殊的初始化和缩放，平衡不同层的学习率。
class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()

        #可变参数大小为(out_dim, in_dim)的张量
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        
        if bias:
            #值全为bias_int的(out_dim, )张量
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None

        #激活函数    
        self.activation = activation
        #缩放因子，来平衡不同层的学习率
        self.scale = (1 / sqrt(in_dim)) * lr_mul
        #学习率乘数
        self.lr_mul = lr_mul
        
    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            if self.bias is not None:
                out = out + self.bias * self.lr_mul
            out = self.activation(out)
        else:
            out = F.linear(input, self.weight * self.scale, 
                           bias=self.bias * self.lr_mul if self.bias is not None else None)
        
        return out

# 映射网络将输入的随机噪声向量 z 映射到风格空间 w，通过depth个 EqualLinear 层实现。
class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim, depth=8):  #depth层数
        super().__init__()
        layers = []
        for i in range(depth):
            layers.append(EqualLinear(
                z_dim if i == 0 else w_dim,
                w_dim,
                lr_mul=0.01,
                activation=nn.LeakyReLU(0.2)
            ))
        #封装成模块
        self.mapping = nn.Sequential(*layers)
        
    def forward(self, z):
        # 归一化潜在编码
        z = F.normalize(z, dim=1)
        w = self.mapping(z)
        return w

# 自适应实例归一化
class AdaIN(nn.Module):
    def __init__(self, feature_dim, style_dim):
        super().__init__()
        #在每个样本的每个通道上独立计算均值和方差，然后进行归一化
        self.instance_norm = nn.InstanceNorm1d(feature_dim)
        #将风格向量映射到一个与输入特征维度相同的向量，用于生成自适应的缩放因子
        self.style_scale = EqualLinear(style_dim, feature_dim)
        #将风格向量映射到一个与输入特征维度相同的向量，用于生成自适应的偏移量
        self.style_bias = EqualLinear(style_dim, feature_dim)
        
    def forward(self, x, style):
        #x:(batch_size, feature_dim, sequence_length)  style:(batch_size, style_dim)
        normalized = self.instance_norm(x)
        scale = self.style_scale(style).unsqueeze(2)        #(batch_size, feature_dim, 1)
        bias = self.style_bias(style).unsqueeze(2)
        transformed = normalized * scale + bias
        return transformed