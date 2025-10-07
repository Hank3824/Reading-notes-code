import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from transformer import Encoder

def gelu(x):
    """GELU activation function"""
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

class PredictModel(nn.Module):
    def __init__(self, num_layers=6, d_model=256, dff=512, num_heads=8, vocab_size=18, a=2, dropout_rate=0.1, dense_dropout=0.1):
        super(PredictModel, self).__init__()

        # 编码器：输出序列级表征；后续取第 0 个 token（CLS）作为全局分子向量 
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                              num_heads=num_heads, dff=dff, input_vocab_size=vocab_size,
                              maximum_position_encoding=200, dropout_rate=dropout_rate)

        # 预测头（CLS 向量 -> 分子性质）：非线性 + dropout + 输出层
        self.fc1 = nn.Linear(d_model, 256)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dense_dropout)
        self.fc2 = nn.Linear(256, a)  # 输出到任务维度 a

    def forward(self, x, adjancecy_matrix, mask=None, training=False):
        # 1) 编码序列，得到逐位置表示
        x = self.encoder(x, mask=mask, adjancecy_matrix=adjancecy_matrix, training=training)
        # 2) 取 CLS 向量（序列第 0 位）作为分子级全局表征
        x = x[:, 0, :]  # Take the first token (CLS token)
        # 3) 通过预测头映射到任务空间
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x



        
