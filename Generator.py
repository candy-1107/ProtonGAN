import torch
import torch.nn as nn
from torch.nn import Sequential
from attention import HierarchicalAttention
from model import AdaIN, MappingNetwork

PROTEIN_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def get_kernel_size(max_seq_length):
    length = 6 * 2**6
    padding = 0
    kernel_size = 0
    while kernel_size <= 0:
        kernel_size = length - max_seq_length + 2 * padding + 1
        padding += 1

    return kernel_size, padding-1



# StyleGAN2生成器块
class StyleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim, kernel_size=3, padding=1,
                 seq_length=None, attention_type='auto'):
        super().__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, padding=padding)
        self.adain = AdaIN(out_channel, style_dim)
        self.activation = nn.LeakyReLU(0.2)

        # 添加稀疏注意力模块
        self.use_attention = seq_length is not None
        if self.use_attention:
            self.attention = HierarchicalAttention(
                out_channel,
                seq_length,
                attention_type=attention_type
            )

            # Style gating 机制 - 可选用风格向量控制注意力强度
            self.attn_gate = nn.Linear(style_dim, 1)

    def forward(self, x, style):
        x = self.conv(x)
        x = self.adain(x, style)
        x = self.activation(x)

        # 应用注意力机制
        if self.use_attention:
            # 将卷积特征转置为 [batch, seq_len, channels]
            x_perm = x.permute(0, 2, 1)

            # 计算注意力输出
            attn_out = self.attention(x_perm)

            # 简化的 gate 实现，避免形状不匹配
            gate_val = torch.sigmoid(self.attn_gate(style))  # [batch, 1]
            gate = gate_val.view(-1, 1, 1)  # 重塑为 [batch, 1, 1]

            x_perm = x_perm + (gate * attn_out)

            # [batch, channels, seq_len]
            x = x_perm.permute(0, 2, 1)

        return x


# 完整的StyleGAN2生成器
class StyleGAN2Generator(nn.Module):
    def __init__(self, z_dim, w_dim, max_seq_length, map_depth=8):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.max_seq_length = max_seq_length

        # 映射网络
        self.mapping = MappingNetwork(z_dim, w_dim, depth=map_depth)

        # 初始常量输入
        self.initial_length = max_seq_length // (2**6)
        self.initial_channels = 512
        self.initial_constant = nn.Parameter(torch.randn(1, self.initial_channels, self.initial_length))

        # 计算每层序列长度
        seq_lengths = [self.initial_length]
        for i in range(6):  # 6次上采样
            seq_lengths.append(seq_lengths[-1] * 2)

        # 为不同层次指定不同注意力类型
        attention_types = [
            'sparse',  # 第一层：稀疏自注意力
            'bigbird',  # 第二层：BigBird
            'performer',  # 第三层：Performer
            'auto'  # 第四层：根据序列长度自动选择
        ]

        # 风格块
        self.style_blocks = nn.ModuleList([
            StyleBlock(512, 512, w_dim),
            StyleBlock(512, 256, w_dim),
            StyleBlock(256, 128, w_dim),
            StyleBlock(128, 64, w_dim),
            StyleBlock(64, 32, w_dim),
            StyleBlock(32, 20, w_dim),
        ])

        # 上采样层
        self.upsamples = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Upsample(scale_factor=2, mode='nearest')
        ])

        #最终输出层
        kernel_size, padding = get_kernel_size(max_seq_length)
        self.final_model = Sequential(
            nn.Conv1d(20, 64, kernel_size=kernel_size, padding=padding, stride=1),
            nn.Conv1d(64, len(PROTEIN_ALPHABET), kernel_size=1)
        )

    def forward(self, z, truncation=0.7, truncation_latent=None):
        # 获取风格编码
        w = self.mapping(z)

        # 截断技巧 (truncation trick)
        if truncation < 1:
            if truncation_latent is None:
                truncation_latent = torch.zeros(1, self.w_dim).to(z.device)
            w = truncation_latent + truncation * (w - truncation_latent)

        # 从常量开始
        batch_size = z.size(0)
        x = self.initial_constant.repeat(batch_size, 1, 1)

        # 逐层处理
        for i, (style_block, upsample) in enumerate(zip(self.style_blocks, self.upsamples)):
            x = upsample(x)
            x = style_block(x, w)

        # 添加适量噪声增加多样性
        noise_scale = 0.05  # 噪声强度
        if self.training:
            noise = torch.randn_like(x) * noise_scale
            x = x + noise

        # 最终输出
        output = self.final_model(x)
        return torch.tanh(output)  # 使用tanh激活