import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from transformer import Encoder

def gelu(x):
    """GELU activation function"""
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

class BertModel(nn.Module):
    def __init__(self, num_layers=6, d_model=256, dff=512, num_heads=8, vocab_size=18, dropout_rate=0.1):
        super(BertModel, self).__init__()

        # 1. 编码器：将原子序列编码为上下文表示
        # Bert模型由transformer的encoder组成
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                            num_heads=num_heads, dff=dff, input_vocab_size=vocab_size,
                            maximum_position_encoding=200, dropout_rate=dropout_rate)

        # 2. 预测头：将编码表示映射到词表
        self.fc1 = nn.Linear(d_model, d_model)
        self.layernorm = nn.LayerNorm(d_model)
        self.fc2 = nn.Linear(d_model, vocab_size)

    def forward(self, x, adjacecy_matrix, mask=None, training=False):
        
        # 步骤1：编码序列
        x = self.encoder(x, mask=mask, adjacecy_matrix=adjacecy_matrix, training=training)

        # 步骤2：前馈变换
        x = self.fc1(x)
        x = gelu(x)

        # 步骤3：层归一化
        x = self.layernorm(x)

        # 步骤4：映射到词表
        x = self.fc2(x)  
        # 输出：每个位置对vocab_size种原子类型的预测概率

        return x



        
