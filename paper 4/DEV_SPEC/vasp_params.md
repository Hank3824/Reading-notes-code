# VASP 参数规范

> 所有计算（CO₂RR 和 OER）使用以下统一参数，确保 LFT 力场可以无缝迁移。
> 对应模板文件：`templates/INCAR_opt`，`templates/INCAR_sp`，`templates/INCAR_freq`

---

## INCAR 基础参数

```fortran
! ── 基础设置 ──────────────────────────────
ISTART = 0          ! 从头开始
ICHARG = 2          ! 初始化电荷密度
ISPIN  = 2          ! 自旋极化（TM 原子磁性，必须开启）
ENCUT  = 500        ! 截断能（eV）— 禁止修改，LFT 要求一致性
PREC   = Accurate   ! 精度

! ── 电子迭代 ──────────────────────────────
EDIFF  = 1E-5       ! 电子收敛标准（eV）
ALGO   = All        ! 算法（金属体系推荐）
NELM   = 200        ! 最大电子步数
NELMIN = 6

! ── 结构优化（INCAR_opt）─────────────────
NSW    = 300        ! 最大离子步数
IBRION = 2          ! CG 算法
ISIF   = 2          ! 固定晶格，优化原子位置
EDIFFG = -0.03      ! 力收敛标准（eV/Å）
POTIM  = 0.5

! ── 单点计算（INCAR_sp，LFT 标注步）──────
NSW    = 0          ! 禁止离子弛豫
IBRION = -1

! ── 频率计算（INCAR_freq，VASP 对照用）──
IBRION = 5          ! 有限差分
NFREE  = 2
POTIM  = 0.015

! ── 输出控制 ──────────────────────────────
LWAVE  = .FALSE.    ! 不写 WAVECAR（节省空间）
LCHARG = .FALSE.    ! 不写 CHGCAR（差分电荷密度分析时改为 .TRUE.）
LORBIT = 11         ! 输出分态密度（DOSCAR），d 带中心分析必须

! ── 色散校正 ──────────────────────────────
IVDW   = 12         ! DFT-D3 BJ 阻尼

! ── 隐式溶剂（VASPsol）────────────────────
LSOL   = .TRUE.
EB_K   = 78.4       ! 水在298K的介电常数（精确值）

! ── 自旋-轨道耦合（可选，仅 5d 重元素）──
! LSORBIT = .TRUE.
```

---

## KPOINTS

使用 VASPKIT 自动生成 Γ 中心网格：

```bash
vaspkit -task 102 -kgamma T -kspacing 0.04
```

对于 4×4 超胞石墨烯 slab，通常得到 **3×3×1** 的 K 点网格。

---

## POTCAR

使用 PAW-PBE 赝势（VASP 5.4.4）：

| 元素 | POTCAR |
|------|--------|
| C    | `C` |
| N    | `N` |
| H    | `H` |
| Fe   | `Fe_pv` |
| Co   | `Co` |
| Ni   | `Ni` |
| Cu   | `Cu_pv` |
| 其他 TM | 优先选含 d 电子版本 |

POTCAR 映射完整表：`templates/POTCAR_map.json`

---

## 气相参考分子计算（Phase 03 前必须完成）

> 禁止使用文献值，必须用当前 VASP 设置自洽计算。

```
CO₂：线性分子，ISPIN=1，不加色散
CO ：ISPIN=1
H₂ ：ISPIN=1，检查轨道占据数
H₂O：ISPIN=1
O₂ ：ISPIN=3（三重态），MAGMOM=2 -2
```

计算结果存入 `data/reference_energies.json`。
