import torch
import torch.nn as nn
import math
from attention import SparseSelfAttention

PROTEIN_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

# StyleGAN2判别器块
class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, downsample=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size, padding=padding),
            nn.LeakyReLU(0.2),
            nn.Conv1d(out_channel, out_channel, kernel_size, padding=padding),
            nn.LeakyReLU(0.2)
        )
        #使用大小为 2 的一维平均池化层，用于下采样
        self.downsample = nn.AvgPool1d(2) if downsample else nn.Identity()
        #跳跃连接，将输入特征直接传递到输出，与经过卷积处理后的特征相加
        self.skip = nn.Conv1d(in_channel, out_channel, kernel_size=1)
        
    def forward(self, x):
        skip = self.skip(x)
        x = self.conv(x)
        x = (x + skip) / math.sqrt(2)  # 残差连接
        x = self.downsample(x)
        return x

# StyleGAN2判别器
class StyleGAN2Discriminator(nn.Module):
    def __init__(self, max_seq_length, sparse_rate=0.3):
        super().__init__()
        
        #将输入的蛋白质序列的独热编码转换为 64 通道的特征图
        self.from_protein = nn.Conv1d(len(PROTEIN_ALPHABET), 64, kernel_size=1)
        
        # 判别器块
        self.blocks = nn.ModuleList([
            DiscriminatorBlock(64, 128),
            DiscriminatorBlock(128, 256),
            DiscriminatorBlock(256, 512),
            DiscriminatorBlock(512, 512, downsample=True)
        ])
        
        # 稀疏自注意力增强
        reduced_seq_length = max_seq_length // 8  # 经过三次降采样
        self.top_k = max(1, int(reduced_seq_length * sparse_rate))
        self.attention = SparseSelfAttention(512, top_k=self.top_k)
        
        # 最终分类器,变成一个概率
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),    #自适应平均池化层
            nn.Flatten(),   # 扁平化层
            nn.Linear(512, 1)   #线性层
        )
        
    def forward(self, x):
        # 初始处理
        x = self.from_protein(x)
        
        # 通过判别器块
        for block in self.blocks:
            x = block(x)
        
        # 特征增强
        x_permuted = x.permute(0, 2, 1)
        x_enhanced, _ = self.attention(x_permuted)
        x_enhanced = x_enhanced.permute(0, 2, 1)
        
        # 分类
        return self.classifier(x_enhanced)