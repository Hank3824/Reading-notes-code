"""
MRL_LIB 工具函数模块
===================

这个模块包含了MRL_LIB项目中用于分子处理、功能基团识别和数据结构转换的核心工具函数。
主要功能包括：
1. 功能基团列表生成
2. SMILES格式转换和标准化
3. 分子功能基团识别
4. 分子图结构转换
5. 数据集处理辅助函数

"""

from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops
import numpy as np
import os
import csv
from typing import List, Dict, Tuple, Optional
from rdkit import RDConfig
from rdkit.Chem import FragmentCatalog

def _validate_smarts(smarts: str) -> bool:
    """
    验证SMARTS字符串是否有效
    
    Args:
        smarts: SMARTS字符串
        
    Returns:
        bool: 如果SMARTS有效返回True，否则返回False
    """
    try:
        mol = Chem.MolFromSmarts(smarts)
        return mol is not None
    except Exception:
        return False


def _print_statistics(stats: Dict[str, int]) -> None:
    """
    打印功能基团统计信息
    
    Args:
        stats: 包含统计信息的字典
    """
    print("=" * 60)
    print("功能基团统计信息 (Functional Groups Statistics):")
    print(f"  RDKit FunctionalGroups.txt: {stats['rdkit_count']}")
    print(f"  RDKit 额外添加: {stats['rdkit_extra_count']}")
    print(f"  moieties.py 定义: {stats['moieties_count']}")
    print(f"  去重前总数: {stats['total_before_dedup']}")
    print(f"  去重后总数: {stats['total_after_dedup']}")
    print(f"  有效SMARTS模式: {stats['valid_count']}")
    print(f"  移除的无效模式: {stats['invalid_count']}")
    print("=" * 60)


def fg_list(validate_smarts: bool = True, verbose: bool = False) -> List[str]:
    """
    生成功能团列表

    从RDKit中的FunctionGroups.txt和moieties.py中提取预定义的功能团

    Return:
        List[str]: 包含所有功能基团的SMARTS字符串的列表

    Note:
        -移除FunctionGroups.txt中索引值为27的功能基团-X (通用卤素模式 *-[#9,#17,#35,#53]), 后续添加更加具体的卤素
    """

    # ===== 第一部分：读取RDKit预定义功能基团 =====

    # 获取RDKit数据目录中功能基团定义文件的路径
    fName = os.path.join(RDConfig.RDDataDir, 'FunctionalGroups.txt')

    # 创建功能基团参数对象，设置最小和最大原子数为1-6
    fparams = FragmentCatalog.FragCatParams(1, 6, fName)

    # 初始化功能基团列表
    rdkit_fg_list = []

    # 遍历所有预定义的功能基团并添加到列表中
    for i in range(fparams.GetNumFuncGroups()):
        rdkit_fg_list.append(fparams.GetFuncGroup(i))

    # 移除第27个功能基团（通用卤素模式 *-[#9,#17,#35,#53]）
    rdkit_fg_list.pop(27)

    # 转换为SMARTS字符串
    rdkit_smarts = [Chem.MolToSmarts(_) for _ in rdkit_fg_list]

    # 并添加具体的卤素和其他的常见功能基团
    rdkit_extra = [
        '*C=C',    # 烯烃
        '*F',      # 氟原子
        '*Cl',     # 氯原子
        '*Br',     # 溴原子
        '*I',      # 碘原子
        '[Na+]',   # 钠离子
        '*P',      # 磷原子
        '*P=O',    # 磷酸基
        '*[Se]',   # 硒原子
        '*[Si]'    # 硅原子
    ]

    # 去重（保持顺序）
    all_fgs = rdkit_smarts + rdkit_extra

    seen = set()
    unique_fgs = []
    for fg in all_fgs:
        if fg not in seen:
            seen.add(fg)
            unique_fgs.append(fg)

    # ===== 第四部分：验证SMARTS有效性（可选） =====
    if validate_smarts:
        valid_fgs = []
        invalid_count = 0
        for smarts in unique_fgs:
            if _validate_smarts(smarts):
                valid_fgs.append(smarts)
            else:
                invalid_count += 1
                if verbose:
                    print(f"Warning: 无效的SMARTS模式: {smarts}")
    else:
        valid_fgs = unique_fgs
        invalid_count = 0

    # ===== 输出统计信息 =====
    if verbose:
        stats = {
            'rdkit_count': len(rdkit_smarts),
            'rdkit_extra_count': len(rdkit_extra),
            'total_before_dedup': len(all_fgs),
            'total_after_dedup': len(unique_fgs),
            'valid_count': len(valid_fgs),
            'invalid_count': invalid_count
        }
        _print_statistics(stats)
    
    return valid_fgs

def rdsmitosmile(smi):
    """
    将SMILES字符串转换为标准化的SMILES格式
    """
    # 使用RDKit进行标准化
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            # 使用RDKit标准化SMILES
            smile = Chem.MolToSmiles(mol, canonical=True)
            return smile
        else:
            # 如果无法解析，返回原始字符串
            print(f"警告: 无法解析SMILES字符串: {smi}")
            return smi
    except Exception as e:
        print(f"RDKit转换失败: {e}")
        return smi

def molecular_fg(smiles):
    """
    识别分子中的功能基团（包括环结构）

    Args:
        smiles (str): 分子的SMILES字符串表示
        
    Returns:
        list: 包含所有功能基团原子索引的列表
              每个元素是一个列表，包含该功能基团中所有原子的索引
              
    Note:
        - 首先识别分子中的环结构(通过SSSR算法)
        - 然后识别预定义的功能基团模式
        - 返回分子中，所有功能基团的原子索引，用于后续的掩码操作
    """


    # 使用RDKit解析SMILES字符串
    mol = Chem.MolFromSmiles(smiles)

    # 获取预定义的功能基团列表
    a = fg_list()

    # 使用RDKit的SSSR（Smallest Set of Smallest Rings）算法识别环结构
    ssr = Chem.GetSymmSSSR(mol)
    num_ring = len(ssr)

    # 创建环结构字典，键为环编号，值为环中原子索引列表
    ring_dict = {}
    for i in range(num_ring):
        ring_dict[i+1] = list(ssr[i])

    # 初始化功能基团列表
    f_g_list = []

    # 将所有环结构添加到功能基团列表中
    for i in ring_dict.values():
        f_g_list.append(i)
    
    # 遍历所有预定义的功能基团模式
    for i in a:
        # 将SMILES字符串转换为SMARTS模式对象
        patt = Chem.MolFromSmarts(i)
        
        # 检查分子是否包含该功能基团模式
        flag = mol.HasSubstructMatch(patt)
        
        if flag:
            # 获取所有匹配的原子索引
            atomids = mol.GetSubstructMatches(patt)
            
            # 将每个匹配的功能基团添加到列表中
            for atomid in atomids:
                f_g_list.append(list(atomid))
    
    return f_g_list


def smiles2adjacecy(smiles, explicit_hydrogens=True, canonical_atom_order=False):
    """
    将SMILES格式的分子转换为原子列表和邻接矩阵
    
    这是FG-BERT模型的核心数据预处理函数, 将分子结构转换为图神经网络可以处理的格式。
    邻接矩阵用于在Transformer的注意力机制中编码分子的拓扑结构信息。
    
    Args:
        smiles (str): 分子的SMILES字符串表示
        explicit_hydrogens (bool, optional): 是否显式包含氢原子. Defaults to True.
        canonical_atom_order (bool, optional): 是否使用标准原子排序. Defaults to False.
        
    Returns:
        tuple: (atoms_list, adjacecy_matrix)
            - atoms_list (list): 原子符号列表，按原子索引顺序排列
            - adjacecy_matrix (numpy.ndarray): 邻接矩阵，形状为(num_atoms, num_atoms)
              
    Note:
        - 邻接矩阵是对称矩阵, 1表示两个原子之间有化学键连接
        - 对角线元素为1(原子与自身连接)
        - 邻接矩阵将在后续处理中转换为注意力掩码
    """


    # 使用RDKit解析SMILES字符串
    mol = Chem.MolFromSmiles(smiles)

    # 根据参数决定是否显式包含氢原子
    if explicit_hydrogens:
        mol = Chem.AddHs(mol)  # 添加氢原子
    else:
        mol = Chem.RemoveHs(mol)  # 移除氢原子
    
    # 根据参数决定是否使用标准原子排序
    if canonical_atom_order:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)

     # 获取分子中的原子数量
    num_atoms = mol.GetNumAtoms()

    # 创建原子符号列表
    atoms_list = []
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        atoms_list.append(atom.GetSymbol())

    # 初始化邻接矩阵（单位矩阵，对角线为1）
    adjacecy_matrix = np.eye(num_atoms)

    # 获取分子中的化学键数量
    num_bonds = mol.GetNumBonds()

    # 遍历所有化学键，更新邻接矩阵
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()  # 化学键起始原子索引
        v = bond.GetEndAtomIdx()    # 化学键终止原子索引

        # 在邻接矩阵中标记两个原子之间的连接（对称矩阵）
        adjacecy_matrix[u, v] = 1.0
        adjacecy_matrix[v, u] = 1.0
    
    return atoms_list, adjacecy_matrix

def get_header(path):
    """
    从CSV文件中读取表头 (第一行)
    
    用于获取数据集的列名信息，这对于理解数据集结构和提取任务标签很重要。
    
    Args:
        path (str): CSV文件的路径
        
    Returns:
        list: 包含CSV文件第一行所有列名的列表
        
    Note:
        - 使用csv.reader读取文件, 确保正确处理CSV格式
        - 只读取第一行作为表头
    """
    with open(path) as f:
        header = next(csv.reader(f))

    return header


# ===== 测试代码 =====
if __name__ == "__main__":
    print("开始测试MRL_LIB工具函数...")
    print("=" * 80)
    
    # 测试1: 基本功能测试
    print("\n1. 测试基本功能基团列表生成:")
    try:
        fg_list_result = fg_list(verbose=False)
        print(f"✓ 成功生成功能基团列表，共 {len(fg_list_result)} 个")
        
        # 显示前10个功能基团
        print("   前10个功能基团示例:")
        for i, fg in enumerate(fg_list_result[:10]):
            print(f"   {i+1:2d}. {fg}")
        
    except Exception as e:
        print(f"✗ 功能基团列表生成失败: {e}")