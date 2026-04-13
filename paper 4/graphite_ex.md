1. 新建石墨烯单层结构：
   Build → Crystals → Build Crystal
   
   晶系：Hexagonal
   a = b = 2.4670 Å   ← 用 a_DFT
   c = 15 Å            ← 现在加真空层
   γ = 120°
   
   原子：
   C1: (0,     0,     0.5)
   C2: (1/3, 2/3,   0.5)

2. 扩展为 6×6 超胞：
   Build → Symmetry → Supercell
   → 6 × 6 × 1
   → 得到  个 C 原子

3. 导出 POSCAR：
   File → Export → POSCAR

# =============================================
# Step 1: Graphene Primitive Cell Optimization
# =============================================

# ----- 系统基本信息 -----
SYSTEM = graphene_primitive

# ----- 初始化设置 -----
ISTART = 0          # 从头开始
ICHARG = 2          # 叠加原子电荷密度作为初始电荷

# ----- 精度与截断能 -----
ENCUT  = 520        # 平面波截断能，eV
PREC   = Accurate   # 精度模式
GGA    = PE         # 显式指定PBE泛函

# ----- 电子步设置 -----
ALGO   = Normal     # 电子步迭代算法（Davidson）
NELM   = 200        # 最大电子步数，默认60有时不够
NELMIN = 6          # 最少电子步数，防止过早退出
EDIFF  = 1E-6       # 电子步收敛标准

# ----- 关键！展宽方法 -----
ISMEAR = 0          # Gaussian展宽
                    # 石墨烯是半金属（Dirac点）
                    # 不能用 ISMEAR=-5（绝缘体方法）
                    # 不建议用 ISMEAR=1（MP展宽，金属用）
SIGMA  = 0.05       # 展宽宽度，eV
                    # 石墨烯用小值0.05，防止虚假态贡献

# ----- 自旋设置 -----
ISPIN  = 1          # 纯碳无磁性，不需要自旋极化

# ----- 离子步设置 -----
IBRION = 2          # 共轭梯度法
NSW    = 100        # 最大离子步数
ISIF   = 3          # 优化原子位置+晶格常数+形状
EDIFFG = -0.001     # 力收敛标准，eV/Å
POTIM  = 0.5        # 离子步步长，默认0.5，CG法适用

# ----- 实空间投影 -----
LREAL  = .FALSE.    # ← 重要！
                    # 原胞只有2个原子，体系极小
                    # 必须用倒空间投影，否则精度损失严重
                    # 大超胞（>50原子）才考虑 LREAL=Auto

# ----- 范德华校正 -----
IVDW   = 12         # 

# ----- 输出控制 -----
LWAVE  = .FALSE.    # 原胞优化不需要保存WAVECAR
LCHARG = .FALSE.    # 不需要保存CHGCAR
LORBIT = 0          # 原胞优化不需要DOS输出

# ----- 并行设置 -----
NCORE  = 4          # 根据计算节点核数调整


用 VASPKIT 102 生成，输入 Rk ≈ 0.03

或者直接手写：
Automatic Gamma
0
Gamma
3  3  1
0  0  0

得到a_DFT = 2.4670！


step2:
用a_DFT = 2.4670调整石墨烯晶格大小，然后进行6×6的超胞


INCAR文件：
# =============================================
# Step 1: 6×6 Graphene Slab Optimization
# =============================================

SYSTEM = graphene_6x6_slab

# ----- 初始化 -----
ISTART = 0
ICHARG = 2

# ----- 精度与截断能 -----
ENCUT  = 520
PREC   = Accurate
GGA    = PE

# ----- 电子步 -----
ALGO   = Normal
NELM   = 200
NELMIN = 6
EDIFF  = 1E-5

# ----- 展宽方法 -----
ISMEAR = 0          # 石墨烯用 Gaussian
SIGMA  = 0.05

# ----- 自旋 -----
ISPIN  = 1          # 纯碳暂不需要自旋

# ----- 离子步 -----
IBRION = 2
NSW    = 200
ISIF   = 2          # ← 只优化原子位置！不动晶格
EDIFFG = -0.02      # 一般为-0.02
POTIM  = 0.5

# ----- 实空间投影 -----
LREAL  = Auto       # 个原子，超胞可以用Auto

# ----- 范德华校正 -----
IVDW   = 12

# ----- 输出 -----
LWAVE  = .TRUE.
LCHARG = .TRUE.
LORBIT = 0

NCORE  = 4
```



