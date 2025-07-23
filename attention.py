import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 添加 BigBird 稀疏注意力机制
class BigBirdAttention(nn.Module):
    def __init__(self, dim, num_heads=8, block_size=64, num_random_blocks=3, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.block_size = block_size
        self.num_random_blocks = num_random_blocks

        # 注意力映射层
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # 投影查询、键、值
        q = self.q_proj(x).view(batch_size, self.num_heads, seq_len, self.head_dim)
        k = self.k_proj(x).view(batch_size, self.num_heads, seq_len, self.head_dim)
        v = self.v_proj(x).view(batch_size, self.num_heads, seq_len, self.head_dim)

        num_blocks = seq_len // self.block_size
        attn_output = self._bigbird_attention(q, k, v, num_blocks)

        # 输出投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        return self.out_proj(attn_output)

    def _standard_attention(self, q, k, v):
        # 标准注意力计算
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return torch.matmul(attn_weights, v)

    def _bigbird_attention(self, q, k, v, num_blocks):
        batch_size, num_heads, seq_len, head_dim = q.shape
        block_size = self.block_size

        # 创建稀疏注意力掩码
        mask = torch.zeros(seq_len, seq_len, device=q.device)

        # 1. 全局注意：所有块都关注第一个和最后一个块
        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, seq_len)
            # 关注首块
            mask[start_idx:end_idx, :block_size] = 1
            # 关注尾块
            last_block_start = max(0, seq_len - block_size)
            mask[start_idx:end_idx, last_block_start:] = 1

        # 2. 窗口注意：每个块关注相邻块
        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, seq_len)

            # 关注自身
            mask[start_idx:end_idx, start_idx:end_idx] = 1

            # 关注前一个块
            if i > 0:
                prev_start = (i - 1) * block_size
                prev_end = i * block_size
                mask[start_idx:end_idx, prev_start:prev_end] = 1

            # 关注后一个块
            if i < num_blocks - 1:
                next_start = (i + 1) * block_size
                next_end = min((i + 2) * block_size, seq_len)
                mask[start_idx:end_idx, next_start:next_end] = 1

        # 3. 随机注意力
        for _ in range(self.num_random_blocks):
            src_block = torch.randint(0, num_blocks, (1,)).item()
            tgt_block = torch.randint(0, num_blocks, (1,)).item()

            src_start = src_block * block_size
            src_end = min((src_block + 1) * block_size, seq_len)
            tgt_start = tgt_block * block_size
            tgt_end = min((tgt_block + 1) * block_size, seq_len)

            mask[src_start:src_end, tgt_start:tgt_end] = 1

        # 应用掩码
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, -1e9)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        return torch.matmul(attn_weights, v)


# 添加 Performer 线性注意力机制
class PerformerAttention(nn.Module):
    def __init__(self, dim, num_features=None, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # 自动调整特征数，确保不超过 head_dim
        self.num_features = min(num_features or self.head_dim, self.head_dim)

        # 投影层
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # 初始化随机投影矩阵
        self.register_buffer("projection_matrix", self._create_projection())

    def _create_projection(self):
        """创建随机正交投影矩阵，确保维度匹配"""
        # 确保特征数量不超过 head_dim
        actual_num_features = min(self.num_features, self.head_dim)

        # 创建随机投影矩阵
        projection = torch.randn(self.num_heads, actual_num_features, self.head_dim)

        # 半正交化以提高稳定性 (使用 torch.linalg.qr 替代 torch.qr)
        for i in range(self.num_heads):
            q, _ = torch.linalg.qr(projection[i].t(), 'reduced')
            projection[i] = q[:actual_num_features].t()

        return projection

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # 投影 Q、K、V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Performer 核心：特征映射
        q_prime = self._kernel_feature_map(q)  # [batch, heads, seq_len, num_features]
        k_prime = self._kernel_feature_map(k)  # [batch, heads, seq_len, num_features]

        # 线性注意力
        kv = torch.matmul(k_prime.transpose(-2, -1), v)  # [batch, heads, num_features, head_dim]
        attn_output = torch.matmul(q_prime, kv)  # [batch, heads, seq_len, head_dim]

        # 输出投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        return self.out_proj(attn_output)

    def _kernel_feature_map(self, x):
        # 随机特征映射，使用 ReLU 核实现
        projection = torch.matmul(x, self.projection_matrix.transpose(-2, -1))
        # 应用正随机特征
        feature_map = F.relu(projection) / math.sqrt(self.num_features)
        return feature_map

# 稀疏自注意力模块
class SparseSelfAttention(nn.Module):
    def __init__(self, input_dim, top_k=None, dropout=0.1):
        super(SparseSelfAttention, self).__init__()

        self.input_dim = input_dim
        self.top_k = top_k
        self.dropout = nn.Dropout(dropout)  # 防止过拟合

        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)
        self.W_v = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # [batch_size, seq_length, input_dim]
        batch_size, seq_length, _ = x.size()

        Q = self.W_q(x)  # [batch_size, seq_length, input_dim]
        K = self.W_k(x)  # [batch_size, seq_length, input_dim]
        V = self.W_v(x)  # [batch_size, seq_length, input_dim]

        # [batch_size, seq_length, seq_length]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.input_dim, dtype=torch.float32))

        # 只保留每行的top-k个值
        if self.top_k is not None and self.top_k < seq_length:
            # [batch_size, seq_length, top_k]
            top_k_values, top_k_index = torch.topk(scores, k=self.top_k, dim=-1)

            # 与scores相同形状的全都为-inf的掩码
            sparse_mask = torch.full_like(scores, float('-inf'))

            # 将top-k值放在对应位置
            sparse_mask.scatter_(-1, top_k_index, top_k_values)
            scores = sparse_mask

        # 计算注意力权重
        attention_weights = nn.functional.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 计算加权和
        output = torch.matmul(attention_weights, V)  # [batch_size, seq_length, input_dim]

        # 通过线性层输出
        output = self.out_proj(output)

        return output, attention_weights

# 层次注意力调度器
class HierarchicalAttention(nn.Module):
    def __init__(self, dim, seq_length, attention_type='auto'):
        super().__init__()
        self.dim = dim
        self.seq_length = seq_length

        # 根据序列长度和位置自动选择注意力类型
        if attention_type == 'auto':
            if seq_length <= 128:
                self.attention = SparseSelfAttention(dim, top_k=seq_length // 2)
            elif seq_length <= 256:
                self.attention = BigBirdAttention(dim, block_size=32)
            else:
                self.attention = PerformerAttention(dim)
        elif attention_type == 'sparse':
            self.attention = SparseSelfAttention(dim, top_k=seq_length // 4)
        elif attention_type == 'bigbird':
            self.attention = BigBirdAttention(dim, block_size=64)
        elif attention_type == 'performer':
            self.attention = PerformerAttention(dim)


    def forward(self, x):
        return self.attention(x)


# # Transformer编码器层
# class TransformerEncoderLayer(nn.Module):
#     def __init__(self, input_dim, top_k=None, dropout=0.1):
#         super(TransformerEncoderLayer, self).__init__()
#         self.sparse_attn = SparseSelfAttention(input_dim, top_k, dropout)
#         self.norm1 = nn.LayerNorm(input_dim)  # 第一个归一化层
#
#         # 位置编码，提高模型对氨基酸顺序的关注
#         self.pos_encoding = nn.Parameter(torch.randn(1, 1000, input_dim))
#
#         # 前馈网络，进行非线性变换
#         self.feed_forward = nn.Sequential(
#             nn.Linear(input_dim, input_dim * 4),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(input_dim * 4, input_dim),
#             nn.Dropout(dropout)
#         )
#         self.norm2 = nn.LayerNorm(input_dim)  # 第二个归一化层
#
#     def forward(self, x):
#         # 添加位置编码，x是[batch_size, seq_length, input_dim]的张量
#         seq_len = x.size(1)
#         x = x + self.pos_encoding[:, :seq_len, :]
#
#         # 获取加权和
#         attn_output, _ = self.sparse_attn(x)
#         x = self.norm1(x + attn_output)
#
#         # 前馈网络，进行非线性变换
#         ff_output = self.feed_forward(x)
#
#         # 残差连接和层归一化
#         output = self.norm2(x + ff_output)
#
#         return output

