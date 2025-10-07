import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def gelu(x):
    """GELU activation function"""
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

# self-attention
def scaled_dot_product_attention(q, k, v, mask=None, adjacecy_matrix=None):

    matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # QK^T, q, k, v: (batch_size, seq_len, d_model)

    dk = torch.tensor(k.size(-1), dtype=torch.float32, device=k.device)
    scaled_attention_logits = matmul_qk / torch.sqrt(dk)  # QK^T / sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  # 屏蔽pading: tokens=['C', 'C', 'O', '<pad>', '<pad>']
    if adjacecy_matrix is not None:
        scaled_attention_logits += adjacecy_matrix  # 结构引导的注意力：将分子结构信息(邻接矩阵)融入到注意力计算中

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)

    output = torch.matmul(attention_weights, v)  # AV, v: (batch_size, seq_len, d_model)
    '''
    attention_weights:
    🔍 可解释性: 理解模型关注什么
    🐛 调试工具: 验证模型行为是否合理
    📈 模型改进: 注意力正则化和蒸馏
    🧪 科学洞察: 发现化学规律和模式
    📊 可视化: 直观展示分子相互作用
    
    '''

    return output, attention_weights

# MultiHeadAttention由多个self-attention组成
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        # 线性层
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):

        x = x.view(batch_size, -1, self.num_heads, self.depth)  # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, depth)
        return x.transpose(1, 2)  # (batch_size, seq_len, num_heads, depth) -> (batch_size, num_heads, seq_len, depth)

    def forward(self, q, k, v, mask=None, adjacecy_matrix=None):

        batch_size = q.size(0)

        q = self.wq(q)  # Q: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        k = self.wv(k)  # K: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        v = self.wv(v)  # V: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # Q: (batch_size, num_heads, seq_len, depth)
        k = self.split_heads(k, batch_size)  # K: (batch_size, num_heads, seq_len, depth)
        v = self.split_heads(v, batch_size)  # V: (batch_size, num_heads, seq_len, depth)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask, adjacecy_matrix
        )

        # 调整维度，变为每个原子对应8个头的输出
        scaled_attention = scaled_attention.transpose(1, 2).contiguous()  # (batch_size, num_heads, seq_len, depth) -> (batch_size, seq_len, num_heads, depth)

        # 拼接所有头的输出
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)

        # 最终线性变化得到输出
        output = self.dense(concat_attention)  # (batch_size, seq_len, d_model)

        return output, attention_weights

class PointWiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dff):
        super(PointWiseFeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, dff)
        self.linear2 = nn.Linear(dff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = gelu(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PointWiseFeedForwardNetwork(d_model, dff)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None, adjacecy_matrix=None, training=True):

        attn_output, attention_weights = self.mha(x, x, x, mask, adjacecy_matrix)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(ffn_output)

        return out2, attention_weights

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, dropout_rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_vocab_size, d_model)

        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, dropout_rate=dropout_rate)
                                         for _ in range(num_layers)])
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None, adjacecy_matrix=None, training=True):

        seq_len = x.size(1)

        if adjacecy_matrix is not None:
            adjacecy_matrix = adjacecy_matrix.unsqueeze(1)  # (batch_size, seq_len, seq_len) -> (batch_size, 1, seq_len, seq_len)

        # 添加embedding 和 position encoding信息，由于是分子，不需要position信息(重要的不是"第几个原子"，而是"原子间的连接关系,不需要序列信息)
        x = self.embedding(x)
        # 嵌入缩放
        x *= math.sqrt(self.d_model)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x, attention_weights = self.enc_layers[i](x, mask, adjacecy_matrix, training)
        return x







       
