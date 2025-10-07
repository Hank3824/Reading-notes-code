# 分子表示学习预训练指南

## 概述
本项目实现了基于功能基团的分子表示学习预训练，使用BERT风格的掩码语言模型(MLM)在ChEMBL数据集上进行预训练。

## 数据集
- **数据路径**: `data/chembl_select_3/chembl_select_3.txt`
- **数据格式**: TSV文件，包含SMILES列
- **数据量**: 约145万分子

## 预训练方法
- **掩码策略**: 随机选择15%的功能基团进行掩码
- **掩码方式**: 90%概率用`<mask>`替换，10%概率保持原样
- **损失计算**: 只计算被掩码位置的交叉熵损失
- **模型架构**: Transformer编码器 + 预测头

## 快速开始

### 1. 基本预训练
```bash
python run_pretrain.py
```

### 2. 自定义参数
```bash
python run_pretrain.py \
    --data_path data/chembl_select_3/chembl_select_3.txt \
    --output_dir ./checkpoints \
    --batch_size 32 \
    --learning_rate 2e-4 \
    --max_epochs 50 \
    --patience 5 \
    --model_name chembl_bert
```

### 3. 直接运行预训练脚本
```bash
python pretrain.py
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_path` | `data/chembl_select_3/chembl_select_3.txt` | 训练数据路径 |
| `--output_dir` | `./checkpoints` | 模型保存目录 |
| `--batch_size` | `32` | 训练批次大小 |
| `--learning_rate` | `2e-4` | 学习率 |
| `--max_epochs` | `50` | 最大训练轮数 |
| `--patience` | `5` | 早停耐心值 |
| `--device` | `auto` | 训练设备 (auto/cpu/cuda) |
| `--model_name` | `chembl_bert` | 模型名称 |

## 训练特性

### 数据集分割
- **训练集**: 80%
- **验证集**: 10% 
- **测试集**: 10%

### 优化策略
- **优化器**: Adam
- **学习率调度**: ReduceLROnPlateau (验证损失不改善时降低学习率)
- **早停机制**: 验证损失连续5个epoch不改善时停止训练

### 模型保存
- **最佳模型**: `bert_weights{model_name}_best.pth` (验证损失最低)
- **最终模型**: `bert_weights{model_name}_final.pth` (训练结束时的模型)
- **检查点**: 每5个epoch保存一次

## 监控训练进度

训练过程中会显示：
- 训练损失和准确率
- 验证损失和准确率
- 当前学习率
- 早停计数
- 每个epoch的耗时

## 使用预训练模型

预训练完成后，可以使用最佳模型进行下游任务：

```python
import torch
from bert import BertModel

# 加载预训练模型
model = BertModel(
    num_layers=6,
    d_model=256,
    dff=512,
    num_heads=8,
    vocab_size=18,
    dropout_rate=0.1
)

# 加载权重
model.load_state_dict(torch.load('checkpoints/bert_weights_chembl_bert_best.pth'))
model.eval()
```

## 注意事项

1. **内存需求**: 建议至少8GB内存，使用GPU可显著加速训练
2. **训练时间**: 在GPU上预计需要数小时到数天，取决于硬件配置
3. **数据预处理**: 确保SMILES格式正确，无效分子会被跳过
4. **早停**: 建议监控验证损失，避免过拟合

## 故障排除

### 常见问题
1. **CUDA内存不足**: 减小batch_size
2. **数据加载失败**: 检查数据文件路径和格式
3. **训练中断**: 检查点会自动保存，可以从中断处继续

### 性能优化
1. 使用GPU加速训练
2. 调整batch_size以充分利用GPU内存
3. 使用多进程数据加载（Windows上建议设为0）
