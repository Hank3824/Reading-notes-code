#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子表示学习预训练启动脚本
使用ChEMBL数据集进行BERT风格的预训练
"""

import os
import sys
import argparse
import torch
import warnings
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description='分子表示学习预训练')
    parser.add_argument('--data_path', type=str, default='data/chembl_select_3/chembl_select_3.txt',
                        help='训练数据路径')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='模型保存目录')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='训练批次大小')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='学习率')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='最大训练轮数')
    parser.add_argument('--patience', type=int, default=5,
                        help='早停耐心值')
    parser.add_argument('--device', type=str, default='auto',
                        help='训练设备 (auto/cpu/cuda)')
    parser.add_argument('--model_name', type=str, default='chembl_bert',
                        help='模型名称')
    
    args = parser.parse_args()
    
    # 设备选择
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.get_device_name()}")
    
    # 检查数据文件
    if not os.path.exists(args.data_path):
        print(f"错误: 数据文件不存在: {args.data_path}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 导入预训练模块
    try:
        from pretrain import main as pretrain_main, ModelConfig
        print("成功导入预训练模块")
    except ImportError as e:
        print(f"导入预训练模块失败: {e}")
        sys.exit(1)
    
    # 修改配置
    config = ModelConfig()
    config.path = args.output_dir
    config.name = args.model_name
    
    print("\n" + "="*60)
    print("开始分子表示学习预训练")
    print("="*60)
    print(f"数据路径: {args.data_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"最大轮数: {args.max_epochs}")
    print(f"早停耐心值: {args.patience}")
    print(f"模型名称: {args.model_name}")
    print("="*60)
    
    # 运行预训练
    try:
        pretrain_main(
            data_path=args.data_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_epochs=args.max_epochs,
            patience=args.patience,
            device_str=args.device,
            model_name=args.model_name,
        )
        print("\n预训练成功完成!")
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
