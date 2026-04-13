# Phase 01 — Slab 模型构建与 DFT 基准

**对应脚本**：`scripts/build/build_slab.py`，`scripts/build/prerelax_slab.py`，`scripts/utils/slurm.py`

---

## 1.1 目标

构建 25–30 个具有代表性的 TM₁-TM₂-N-C Slab 模型，完成高精度 VASP 基准计算，筛掉热力学不稳定结构，同时产出 LFT 微调的初始训练数据。

---

## 1.2 金属组合选择

**采样方法**：拉丁超立方采样（LHS），见 `scripts/active_learning/sample_initial_combinations.py`

**采样空间维度**：
- 维度 1：d 电子数（范围：1–10）
- 维度 2：Pauling 电负性（范围：1.0–2.4）

**覆盖要求**：
- 3d 过渡金属：Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn（各时期各选 2–3 个）
- 4d 过渡金属：Y, Zr, Nb, Mo, Ru, Rh, Pd, Ag, Cd（选 3–4 个）
- 5d 过渡金属：Hf, Ta, W, Re, Os, Ir, Pt, Au（选 3–4 个）
- NiZn-N-C 作为验证基准（强制包含）

**总数**：25–30 组

**输出文件**：`data/sampled_combinations.csv`

CSV 列说明：

| 列 | 含义 | 备注 |
|---|---|---|
| `index` | 序号（从 1 开始） | — |
| `TM1` / `TM2` | 金属元素符号 | — |
| `name` | 体系标签，如 `NiZn-N-C` | — |
| `d_sum` | TM₁ + TM₂ 的 d 电子数之和（范围 2–20） | 仅供采样质量检验，不参与 DFT 计算 |
| `eneg_sum` | TM₁ + TM₂ 的 Pauling 电负性之和（范围 ~2.4–5.1） | 仅供采样质量检验，不参与 DFT 计算 |
| `row_pair` | 周期组合，如 `3d-4d` | — |

---

## 1.3 Slab 模型结构规范

```
基底：石墨烯单层，6×6 超胞
缺陷：双空位（Divacancy），嵌入 TM₁-TM₂，4 个 N 原子配位
真空层：≥ 15 Å（沿 z 方向）
原子数：1×TM₁ + 1×TM₂ + 62×C + 6×N = 70 个原子
磁化：MAGMOM 根据 TM 种类自动设置（见 scripts/utils/constants.py MAGMOM_CONFIG）
```

---

## 1.4 三步工作流

```
Step 1  build_slab.py        →  data/structures/slabs/{system_id}/POSCAR   (初始猜测)
Step 2  prerelax_slab.py     →  data/structures/slabs/{system_id}/POSCAR   (CHGNet 预弛豫，覆盖)
                             →  data/structures/slabs/{system_id}/INCAR
                             →  data/structures/slabs/{system_id}/KPOINTS
Step 3  VASP on HPC          →  手动上传目录 + 添加 POTCAR → 提交作业
                             →  下载 CONTCAR 到 data/structures/slabs/{system_id}/CONTCAR
```

### Step 2 详细说明：CHGNet 预弛豫

在将结构提交 VASP 精算前，先用 CHGNet 通用预训练力场做快速预弛豫，将初始猜测推到势能面近极小处，减少 VASP 所需离子步数。

- **优化器**：FIRE（原子位置，不优化晶胞）
- **收敛判据**：fmax < 0.05 eV/Å
- **晶格处理**：弛豫过程中晶胞固定，弛豫后用原始晶格矩阵覆盖，防止数值漂移
- **输出检查**：自动记录石墨烯 C 层 z 标准差（< 0.05 Å 为正常）和 TM 原子相对石墨烯的 z 偏移

```bash
# 单体系
python scripts/build/prerelax_slab.py --tm1 Ni --tm2 Zn

# 批量（推荐）
python scripts/build/prerelax_slab.py --csv data/sampled_combinations.csv

# 模板改后只重生成 INCAR/KPOINTS，不重跑 CHGNet
python scripts/build/prerelax_slab.py --csv data/sampled_combinations.csv --regen-incar
```

预弛豫日志写入 `data/structures/slabs/prerelax_log.csv`，含收敛状态、步数、最终 fmax、C 层平整度。

### Step 3 VASP 目录内容

上传 HPC 前，`data/structures/slabs/{system_id}/` 应包含：

| 文件 | 来源 |
|---|---|
| `POSCAR` | Step 2 输出（CHGNet 预弛豫后） |
| `INCAR` | Step 2 输出（`INCAR_opt` 模板） |
| `KPOINTS` | Step 2 输出（Gamma 3×3×1） |
| `POTCAR` | 手动从 PAW 库拼接 |

---

## 1.5 形成能计算

```
E_f = E(TM₁-TM₂-N-C) - E(石墨烯双空位-4N) - μ(TM₁) - μ(TM₂)
```

化学势参考态：
- `μ(TM)`：TM 单质最稳定相的 DFT 总能量（体相）
- `μ(N)`：来自 N₂ 气相分子（½ × E(N₂)）

**筛选标准**：
- `E_f < 0`：热力学稳定，保留，进入后续计算
- `E_f > 0`：热力学不稳定，淘汰

> ⚠ 此步骤不走 LFT，必须使用完整 VASP 精算，是所有后续计算的基础。

---

## 1.6 目录结构

```
data/structures/slabs/
├── NiZn_NC/
│   ├── POSCAR          ← Step 2 输出（CHGNet 预弛豫）/ VASP 输入
│   ├── INCAR           ← Step 2 输出
│   ├── KPOINTS         ← Step 2 输出
│   ├── structure.json  ← Step 1 输出（初始结构备份）
│   └── CONTCAR         ← VASP 精算完成后从 HPC 下载
├── FeCo_NC/
│   └── ...
└── prerelax_log.csv    ← CHGNet 预弛豫汇总日志
```
