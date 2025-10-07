"""
FG-BERT PyTorch 预训练脚本
========================

这个脚本实现了FG-BERT模型的PyTorch版本预训练。
主要功能：
1. 加载分子数据集
2. 进行功能基团掩码预训练
3. 保存预训练权重

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import time
import os
from bert import BertModel
from utils import smiles2adjacecy, molecular_fg
from rdkit import Chem
from rdkit import RDLogger

# 禁用RDKit的警告信息
RDLogger.DisableLog('rdApp.*')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 模型配置
class ModelConfig:
    def __init__(self):
        self.name= 'Medium'
        self.num_layers = 6
        self.num_heads = 4
        self.d_model = 256
        self.dff = self.d_model * 2
        self.vocab_size = 18
        self.dropout_rate = 0.1
        self.addH = True
        self.path = 'medium3_weights_pytorch'

        # 创建权重保存目录
        os.makedirs(self.path, exist_ok=True)

# 原子到数字的映射字典
str2num = {
    '<pad>': 0,     # 填充token
    'H': 1, 'C': 2, 'N': 3, 'O': 4, 'S': 5, 'F': 6, 'Cl': 7, 
    'Br': 8, 'P': 9, 'I': 10, 'Na': 11, 'B': 12, 'Se': 13, 'Si': 14, 
    '<unk>': 15,    # 未知原子
    '<mask>': 16,   # 掩码token（用于预训练）
    '<global>': 17  # 全局token（类似<cls>）
}  # 分子不需要分隔符，没有<sep>

num2str = {i: j for j, i in str2num.items()}

class FunctionalGroupDataset(Dataset):

    def __init__(self, smiles_list, config):
        """
        初始化数据集
        
        Args:
            smiles_list: SMILES字符串列表
            config: 模型配置对象
        """
        self.smiles_list = smiles_list
        self.config = config
        self.str2num = str2num
        self.num2str = num2str

        print(f"数据集大小: {len(self.smiles_list)}")

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        """
        获取单个训练样本
        
        这是预训练的核心函数，执行以下步骤：
        1. 解析SMILES字符串
        2. 识别功能基团
        3. 生成掩码
        4. 构建邻接矩阵
        5. 返回训练数据
        
        Args:
            idx: 样本索引
            
        Returns:
            tuple: (input_ids, adjacecy_matrix, target_ids, mask_weights)
        """
        smiles = self.smiles_list[idx]

        # 步骤1: 识别功能基团
        # 返回分子中所有功能基团的原子索引列表
        f_g_list = molecular_fg(smiles)

        # 步骤2: 将SMILES转换为原子列表和邻接矩阵
        atoms_list, adjacecy_matrix = smiles2adjacecy(smiles, explicit_hydrogens=self.config.addH)

        # 步骤3: Tokenization，<global> token用于聚合整个分子的信息
        atom_list = ['<global>'] + atoms_list
        nums_list = [self.str2num['<global>']] + [self.str2num.get(atom, self.str2num['<unk>']) for atom in atoms_list]

        # 步骤4: 构建注意力掩码
        # 邻接矩阵用于在注意力计算中编码分子结构信息
        temp = np.ones((len(nums_list), len(nums_list)))  # 构建一个大小为nums_list * nums_list, 全是1的矩阵
        temp[1:, 1:] = adjacecy_matrix  # 从第一行第一列开始，第0行第0列表示<global>与所有的原子都相连
        adjacecy_matrix = (1 - temp) * (-1e9)  # 将不连接的原子对设为负无穷

        # 步骤5: 生成掩码训练样本
        # 随机选择15%的功能基团进行掩码
        if len(f_g_list) > 1:
            choices = np.random.permutation(len(f_g_list) - 1)[:max(int(len(f_g_list) * 0.15), 1)] + 1
        else:
            choices = []

        # 创建目标序列和掩码权重
        y = np.array(nums_list).astype('int64')
        weight = np.zeros(len(nums_list))

        # 应用掩码策略
        for i in choices:
            rand = np.random.rand()
            if rand < 0.9:  # 90%概率用<mask>替换
                for j in f_g_list[i]:
                    weight[j] = 1  # 标记需要预测的位置
                    nums_list[j] = self.str2num['<mask>']  # 用掩码token替换
            else:  # 10%概率保持原样
                for j in f_g_list[i]:
                    weight[j] = 1  # 仍然需要预测

        # 转换为张量
        x = np.array(nums_list).astype('int64')
        weight = weight.astype('float32')

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(adjacecy_matrix, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(weight, dtype=torch.float32)
        )

def collate_fn(batch):
    """
    批处理函数
    
    将多个样本组合成一个批次，处理不同长度的序列
    
    Args:
        batch: 批次数据列表
        
    Returns:
        tuple: 批处理后的张量
    """
    # 分离各个组件
    input_ids = [item[0] for item in batch]
    adjacecy_matrices = [item[1] for item in batch]
    target_ids = [item[2] for item in batch]
    mask_weights = [item[3] for item in batch]
    
    # 计算最大序列长度
    max_len = max(len(seq) for seq in input_ids)
    
    # 填充到相同长度
    padded_input_ids = []
    padded_adjoin_matrices = []
    padded_target_ids = []
    padded_mask_weights = []
    
    for i in range(len(batch)):
        seq_len = len(input_ids[i])
        
        # 填充input_ids
        padded_input = torch.cat([
            input_ids[i],
            torch.zeros(max_len - seq_len, dtype=torch.long)
        ])
        padded_input_ids.append(padded_input)
        
        # 填充adjoin_matrix
        padded_adj = torch.zeros(max_len, max_len, dtype=torch.float32)
        padded_adj[:seq_len, :seq_len] = adjacecy_matrices[i]
        padded_adjoin_matrices.append(padded_adj)
        
        # 填充target_ids
        padded_target = torch.cat([
            target_ids[i],
            torch.zeros(max_len - seq_len, dtype=torch.long)
        ])
        padded_target_ids.append(padded_target)
        
        # 填充mask_weights
        padded_weight = torch.cat([
            mask_weights[i],
            torch.zeros(max_len - seq_len, dtype=torch.float32)
        ])
        padded_mask_weights.append(padded_weight)
    
    # 堆叠成批次张量
    return (
        torch.stack(padded_input_ids),
        torch.stack(padded_adjoin_matrices),
        torch.stack(padded_target_ids),
        torch.stack(padded_mask_weights)
    )

def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    训练一个epoch
    
    Args:
        model: 模型
        dataloader: 数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
        
    Returns:
        tuple: (平均损失, 平均准确率)
    """
    model.train()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0

    for batch_idx, (input_ids, adjacecy_matrices, target_ids, mask_weights) in enumerate(dataloader):
         # 移动到设备
        input_ids = input_ids.to(device)
        adjacecy_matrices = adjacecy_matrices.to(device)
        target_ids = target_ids.to(device)
        mask_weights = mask_weights.to(device)

        # 创建注意力掩码（padding mask）
        # 这里创建的是padding mask，用于忽略填充的token
        seq_mask = (input_ids == 0).float()  # 0是<pad>token
        seq_mask = seq_mask.unsqueeze(1).unsqueeze(2)  # 扩展维度用于广播

        # 前向传播
        optimizer.zero_grad()

        # 模型预测
        predictions = model(input_ids, adjacecy_matrix=adjacecy_matrices, mask=seq_mask, training=True)

         # 计算损失
        # 只计算被掩码位置的损失
        loss = criterion(predictions.view(-1, predictions.size(-1)), target_ids.view(-1))

        # 应用掩码权重
        mask_weights_flat = mask_weights.view(-1)
        loss = (loss * mask_weights_flat).sum() / mask_weights_flat.sum()

        # 反向传播
        loss.backward()
        optimizer.step()

        # 计算准确率
        with torch.no_grad():
            predicted = torch.argmax(predictions, dim=-1)
            correct = (predicted == target_ids).float()
            accuracy = (correct * mask_weights).sum() / mask_weights.sum()
        
        total_loss += loss.item()
        total_accuracy += accuracy.item()
        num_batches += 1
        
        # 打印进度
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}')
    
    return total_loss / num_batches, total_accuracy / num_batches

def evaluate(model, dataloader, criterion, device):
    """
    评估模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备
        
    Returns:
        tuple: (平均损失, 平均准确率)
    """
    model.eval()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0
    
    with torch.no_grad():
        for input_ids, adjacecy_matrices, target_ids, mask_weights in dataloader:
            # 移动到设备
            input_ids = input_ids.to(device)
            adjacecy_matrices = adjacecy_matrices.to(device)
            target_ids = target_ids.to(device)
            mask_weights = mask_weights.to(device)
            
            # 创建注意力掩码
            seq_mask = (input_ids == 0).float()
            seq_mask = seq_mask.unsqueeze(1).unsqueeze(2)
            
            # 前向传播
            predictions = model(input_ids, adjacecy_matrix=adjacecy_matrices, mask=seq_mask, training=False)
            
            # 计算损失
            loss = criterion(predictions.view(-1, predictions.size(-1)), target_ids.view(-1))
            mask_weights_flat = mask_weights.view(-1)
            loss = (loss * mask_weights_flat).sum() / mask_weights_flat.sum()
            
            # 计算准确率
            predicted = torch.argmax(predictions, dim=-1)
            correct = (predicted == target_ids).float()
            accuracy = (correct * mask_weights).sum() / mask_weights.sum()
            
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            num_batches += 1
    
    return total_loss / num_batches, total_accuracy / num_batches

def main(
    data_path: str = None,
    output_dir: str = None,
    batch_size: int = None,
    learning_rate: float = None,
    max_epochs: int = None,
    patience: int = None,
    device_str: str = None,
    model_name: str = None,
):
    """
    主训练函数
    
    执行完整的预训练流程：
    1. 加载数据
    2. 创建模型
    3. 训练循环
    4. 保存权重
    """
    print("=" * 60)
    print("FG-BERT PyTorch 预训练开始")
    print("=" * 60)

    # 配置
    config = ModelConfig()
    if output_dir is not None:
        config.path = output_dir
        os.makedirs(config.path, exist_ok=True)
    if model_name is not None:
        config.name = model_name
    
    # 设备
    local_device = None
    if device_str is not None:
        if device_str == 'auto':
            local_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            local_device = torch.device(device_str)
        print(f"(override) 使用设备: {local_device}")

    # 加载数据
    print("步骤1: 加载数据...")
    # 加载ChEMBL数据集
    data_path = data_path or 'data/chembl_select_3/chembl_select_3.txt'
    try:
        if os.path.exists(data_path):
            print(f"正在加载数据集: {data_path}")
            df = pd.read_csv(data_path, sep='\t')
            smiles_field = 'Smiles'
            smiles_list = df[smiles_field].tolist()
            print(f"成功加载了 {len(smiles_list)} 个分子")
        else:
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("使用示例数据...")
        smiles_list = ['CCO', 'CC(=O)O', 'c1ccccc1', 'CCN', 'CC(C)O'] * 100

    
    # 创建数据集
    print("步骤2: 创建数据集...")
    dataset = FunctionalGroupDataset(smiles_list, config)
    
    # 分割数据集 (训练:验证:测试 = 8:1:1)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"数据集分割: 训练集 {train_size}, 验证集 {val_size}, 测试集 {test_size}")

    # 创建数据加载器
    effective_batch_size = batch_size or 32
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0  # Windows上建议设为0
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=max(1, effective_batch_size * 2), 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=max(1, effective_batch_size * 2), 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )

    # 创建模型
    print("步骤3: 创建模型...")
    model = BertModel(
        num_layers=config.num_layers,
        d_model=config.d_model,
        dff=config.dff,
        num_heads=config.num_heads,
        vocab_size=config.vocab_size,
        dropout_rate=config.dropout_rate
    ).to(local_device or device)

     # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 创建优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=(learning_rate))
    criterion = nn.CrossEntropyLoss(reduction='none')  # 不使用reduction，手动应用权重
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # 早停机制
    best_val_loss = float('inf')
    patience = patience or 5
    patience_counter = 0
    best_epoch = 0
    
    # 训练循环
    print("步骤4: 开始训练...")
    num_epochs = max_epochs or 50

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)
        
        start_time = time.time()
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, local_device or device)
        
        # 验证
        val_loss, val_acc = evaluate(model, val_loader, criterion, local_device or device)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        print(f"Epoch时间: {epoch_time:.2f}秒")
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            # 保存最佳权重
            best_weight_path = os.path.join(config.path, f'bert_weights{config.name}_best.pth')
            torch.save(model.state_dict(), best_weight_path)
            print(f"新的最佳模型已保存到: {best_weight_path}")
        else:
            patience_counter += 1
            print(f"验证损失未改善，早停计数: {patience_counter}/{patience}")
        
        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(config.path, f'bert_weights{config.name}_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"检查点已保存到: {checkpoint_path}")
        
        # 早停检查
        if patience_counter >= patience:
            print(f"\n早停触发！最佳epoch: {best_epoch}, 最佳验证损失: {best_val_loss:.4f}")
            break
    
    print("\n" + "=" * 60)
    print("预训练完成！")
    print(f"最佳epoch: {best_epoch}, 最佳验证损失: {best_val_loss:.4f}")
    print("=" * 60)
    
    # 加载最佳模型进行最终测试
    best_model_path = os.path.join(config.path, f'bert_weights{config.name}_best.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("已加载最佳模型进行最终测试")
    
    # 最终测试
    test_loss, test_acc = evaluate(model, test_loader, criterion, local_device or device)
    print(f"最终测试损失: {test_loss:.4f}, 最终测试准确率: {test_acc:.4f}")
    
    # 保存最终模型
    final_model_path = os.path.join(config.path, f'bert_weights{config.name}_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型已保存到: {final_model_path}")

if __name__ == "__main__":
    main()

    