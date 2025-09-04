import os
import pandas as pd
from mp_api.client import MPRester
from rdkit import Chem

# 配置
# API_KEY = "oZGg3Ye.........xZA6volqI"  # 替换为你的 API 密钥

# 输入文件路径
nelements_file_path = r"..\nelements=4.csv"

# 输出文件路径
output_folder = r'..\nelements smiles redox'
output_file = os.path.join(output_folder, 'nelements=4_simles_redox.csv')

# 读取上传的 CSV 文件
df = pd.read_csv(nelements_file_path)
molecule_ids = df['molecule_id'].tolist()

# 初始化变量
batch_size = 1000
total_molecules = len(molecule_ids)
results = []
first_batch = True

with MPRester(api_key="oZGg3YeOT..........xZA6volqI") as mpr:
    for idx, mol_id in enumerate(molecule_ids, 1):
        try:
            # 获取分子数据
            docs = mpr.molecules.summary.search(
                molecule_ids=[mol_id],
                fields=["inchi", "redox"]
            )
            
            smiles = None
            ox_pot_li = None
            
            # 处理每个文档（通常每个mol_id对应一个文档）
            for doc in docs:
                # InChI 转换为 SMILES
                if hasattr(doc, 'inchi') and doc.inchi:
                    mol = Chem.MolFromInchi(doc.inchi)
                    if mol:
                        smiles = Chem.MolToSmiles(mol)
                
                # 获取氧化电位
                if hasattr(doc, 'redox') and doc.redox:
                    for solvent, redox_comp in doc.redox.items():
                        if solvent == "NONE":
                            ox_pot_h = redox_comp.oxidation_potential
                            if ox_pot_h is not None:
                                ox_pot_li = ox_pot_h - 3.04
                break  # 通常只需要处理第一个文档
            
            # 添加结果
            results.append({
                'molecule_id': mol_id,
                'SMILES': smiles,
                'Oxidation Potential (vs Li/Li+)': ox_pot_li
            })

            # 进度提示（每100个或最后显示一次）
            if idx % 100 == 0 or idx == total_molecules:
                print(f"已处理 {idx}/{total_molecules} 个分子（{idx/total_molecules:.1%}）")

            # 分批写入（每1000个或最后写入一次）
            if idx % batch_size == 0 or idx == total_molecules:
                if results:
                    # 确定写入模式和header
                    mode = 'w' if first_batch else 'a'
                    header = first_batch
                    
                    # 写入CSV
                    pd.DataFrame(results).to_csv(
                        output_file,
                        mode=mode,
                        header=header,
                        index=False
                    )
                    
                    # 重置状态
                    results = []
                    first_batch = False
                    print(f"成功写入 {idx} 个分子的数据到文件")

        except Exception as e:
            print(f"处理分子 {mol_id} 时发生错误: {str(e)}")
            continue

print("✅ 处理完成！所有数据已保存")