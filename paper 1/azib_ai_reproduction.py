"""
AI-Driven Electrolyte Additive Selection - Complete Reproduction
基于论文的完整AI流程复现
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(3407)

# Step 1: 生成虚拟数据集
print("="*60)
print("Step 1: 生成虚拟分子数据集")
print("="*60)

def generate_molecular_data(n_samples=38):
    """
    生成虚拟的分子特征数据
    包含17个物理特征（如论文中Table 1所示）
    """
    data = {
        'Molecule_ID': [f'Mol_{i+1}' for i in range(n_samples)],
        'M': np.random.uniform(30, 200, n_samples),  # 分子量 (kg/mol)
        'alpha': np.random.uniform(2, 20, n_samples),  # 极化率 (Å³)
        '#H': np.random.randint(2, 20, n_samples),  # 氢原子数
        '#O': np.random.randint(0, 8, n_samples),  # 氧原子数
        '#HA': np.random.randint(3, 15, n_samples),  # 重原子数
        '#A': np.random.randint(5, 30, n_samples),  # 总原子数
        'ne': np.random.randint(10, 100, n_samples),  # 价电子数
        'HBD': np.random.randint(0, 5, n_samples),  # 氢键供体
        'HBA': np.random.randint(0, 8, n_samples),  # 氢键受体
        'AVE_PE': np.random.uniform(2.0, 3.5, n_samples),  # 平均Pauling电负性
        'AVE_IP': np.random.uniform(5, 15, n_samples),  # 平均离子势 (eV)
        'AVE_EA': np.random.uniform(0.5, 3, n_samples),  # 平均电子亲和能 (eV)
        'sigma': np.random.uniform(20, 80, n_samples),  # 表面张力 (N/m)
        'MP': np.random.uniform(150, 400, n_samples),  # 熔点 (K)
        'BP': np.random.uniform(300, 500, n_samples),  # 沸点 (K)
        'rho': np.random.uniform(700, 1500, n_samples),  # 密度 (kg/m³)
    }
    
    # 计算 #HA/#H 比率
    data['#HA/#H'] = data['#HA'] / data['#H']
    
    df = pd.DataFrame(data)
    
    # 生成表面自由能 (γ) - 目标变量
    # 基于SISSO论文中的关键特征：#HA 和 sigma
    # γ 越负表示越稳定
    gamma = (-0.01 * df['#HA'] - 0.002 * df['sigma'] + 
             0.05 * df['HBA'] - 0.03 * df['AVE_IP'] + 
             np.random.normal(0, 0.05, n_samples))
    
    df['gamma'] = gamma
    
    # 生成库伦效率 (用于验证)
    # 与gamma呈二次关系
    df['CE'] = 95 + 10 * gamma + 5 * gamma**2 + np.random.normal(0, 2, n_samples)
    df['CE'] = np.clip(df['CE'], 70, 99.9)
    
    return df

# 生成训练数据和预测数据
train_data = generate_molecular_data(38)
test_data = generate_molecular_data(27)

print(f"训练集大小: {len(train_data)} 个分子")
print(f"测试集大小: {len(test_data)} 个分子")
print("\n前5个分子的数据:")
print(train_data[['Molecule_ID', '#HA', 'sigma', 'gamma', 'CE']].head())

# Step 2: 验证描述符与实验结果的相关性
print("\n" + "="*60)
print("Step 2: 验证表面自由能与库伦效率的相关性")
print("="*60)

# 二次多项式拟合
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(train_data[['gamma']])
model_validation = LinearRegression()
model_validation.fit(X_poly, train_data['CE'])

# 计算R²
r2_score = model_validation.score(X_poly, train_data['CE'])
print(f"二次多项式拟合 R²: {r2_score:.3f}")

# 绘制关系图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 散点图和拟合曲线
gamma_range = np.linspace(train_data['gamma'].min(), train_data['gamma'].max(), 100)
gamma_poly = poly.transform(gamma_range.reshape(-1, 1))
ce_pred = model_validation.predict(gamma_poly)

ax1.scatter(train_data['gamma'], train_data['CE'], alpha=0.6, s=50)
ax1.plot(gamma_range, ce_pred, 'r-', linewidth=2, label='二次拟合')
ax1.set_xlabel('γ (eV/Å²)')
ax1.set_ylabel('CE (%)')
ax1.set_title('γ vs CE')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 对比：生成假的吸附能数据显示较差相关性
fake_Ead = -0.5 * train_data['M'] / 100 + np.random.normal(0, 0.3, len(train_data))
ax2.scatter(fake_Ead, train_data['CE'], alpha=0.6, s=50, color='orange')
ax2.set_xlabel('Ead (eV)')
ax2.set_ylabel('CE (%)')
ax2.set_title('Ead vs CE')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Step 3: SISSO机器学习预测
print("\n" + "="*60)
print("Step 3: SISSO启发的特征选择和线性回归")
print("="*60)

# 准备特征
feature_cols = ['M', 'alpha', '#H', '#O', '#HA', '#A', 'ne', 'HBD', 'HBA', 
                'AVE_PE', 'AVE_IP', 'AVE_EA', 'sigma', 'MP', 'BP', 'rho', '#HA/#H']

X_train = train_data[feature_cols]
y_train = train_data['gamma']

# 特征重要性分析（使用相关系数模拟SISSO的特征选择）
correlations = {}
for col in feature_cols:
    correlations[col] = abs(stats.pearsonr(X_train[col], y_train)[0])

# 排序并选择最重要的特征
sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
print("\n特征重要性排序（与γ的相关性）:")
for i, (feat, corr) in enumerate(sorted_features[:6], 1):
    print(f"{i}. {feat}: {corr:.3f}")

# 选择top特征构建SISSO模型
top_features = [feat[0] for feat in sorted_features[:4]]
print(f"\n选择的关键特征: {top_features}")

# 构建预测模型
X_selected = X_train[top_features]
model_sisso = LinearRegression()  # SISSO 需要在linux系统上运行，这里用线性回归代替
model_sisso.fit(X_selected, y_train)

# 5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for train_idx, val_idx in kf.split(X_selected):
    X_tr, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    model_temp = LinearRegression()
    model_temp.fit(X_tr, y_tr)
    score = model_temp.score(X_val, y_val)
    cv_scores.append(score)

print(f"5折交叉验证 R²: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

# 预测测试集
X_test = test_data[top_features]
test_data['gamma_pred'] = model_sisso.predict(X_test)
print(f"\n测试集预测完成，RMSE: {np.sqrt(np.mean((test_data['gamma'] - test_data['gamma_pred'])**2)):.3f}")

# Step 4: 特征相关性热图
print("\n" + "="*60)
print("Step 4: 特征相关性分析")
print("="*60)

# 计算相关矩阵
corr_matrix = X_train.corr()

# 绘制热图
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Pearson correlation matrix of features')
plt.tight_layout()
plt.show()

# Step 5: 降维和聚类分析
print("\n" + "="*60)
print("Step 5: 随机树嵌入(RTE)降维 + K-means聚类")
print("="*60)

# 合并训练和测试数据
all_data = pd.concat([train_data, test_data], ignore_index=True)
X_all = all_data[feature_cols]

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# 随机树嵌入降维
print("执行随机树嵌入降维...")
rte = RandomTreesEmbedding(n_estimators=120, max_depth=3, random_state=6)
X_rte = rte.fit_transform(X_scaled)

# 使用t-SNE进一步降到2维（因为RTE输出是高维稀疏的）
tsne = TSNE(n_components=2, random_state=42, perplexity=15)
X_2d = tsne.fit_transform(X_rte.toarray())

all_data['X_embed'] = X_2d[:, 0]
all_data['Y_embed'] = X_2d[:, 1]

# 确定最优聚类数（轮廓系数）
silhouette_scores = []
K_range = range(2, 16)

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_temp = kmeans_temp.fit_predict(X_2d)
    score = silhouette_score(X_2d, labels_temp)
    silhouette_scores.append(score)

optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"最优聚类数 K = {optimal_k} (轮廓系数: {max(silhouette_scores):.3f})")

# 但根据论文，使用K=9
K = 9
kmeans = KMeans(n_clusters=K, random_state=4, n_init=10)
all_data['cluster'] = kmeans.fit_predict(X_2d)

# Step 6: 可视化聚类结果
print("\n" + "="*60)
print("Step 6: 可视化聚类结果和最优区域识别")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 左图：训练集
train_mask = np.arange(len(train_data))
ax1 = axes[0]
scatter1 = ax1.scatter(all_data.iloc[train_mask]['X_embed'], 
                       all_data.iloc[train_mask]['Y_embed'],
                       c=all_data.iloc[train_mask]['gamma'], 
                       cmap='RdBu_r', s=100, alpha=0.7, edgecolors='black')

ax1.set_title('train molecules distribution (color=γ)')
ax1.set_xlabel('RTE, x')
ax1.set_ylabel('RTE, y')
plt.colorbar(scatter1, ax=ax1, label='γ (eV/Å²)')

# 添加Voronoi边界
from scipy.spatial import Voronoi, voronoi_plot_2d
vor = Voronoi(kmeans.cluster_centers_)
voronoi_plot_2d(vor, ax=ax1, show_vertices=False, line_colors='gray', 
                line_width=1, line_alpha=0.3, point_size=0)

# 右图：所有数据带聚类
ax2 = axes[1]
scatter2 = ax2.scatter(all_data['X_embed'], all_data['Y_embed'],
                       c=all_data['gamma'], cmap='RdBu_r', 
                       s=100, alpha=0.7, edgecolors='black')

# 标记测试集
test_mask = np.arange(len(train_data), len(all_data))
ax2.scatter(all_data.iloc[test_mask]['X_embed'], 
           all_data.iloc[test_mask]['Y_embed'],
           c='none', s=150, edgecolors='red', linewidths=2, 
           label='test molecules')

ax1.set_title('test molecules distribution (color=γ)')
ax2.set_xlabel('RTE, x')
ax2.set_ylabel('RTE, y')
ax2.legend()
plt.colorbar(scatter2, ax=ax2, label='γ (eV/Å²)')

plt.tight_layout()
plt.show()

# Step 7: 识别最优区域
print("\n" + "="*60)
print("Step 7: 识别最优添加剂区域")
print("="*60)

# 计算每个聚类的平均γ值
cluster_stats = all_data.groupby('cluster')['gamma'].agg(['mean', 'std', 'count'])
cluster_stats = cluster_stats.sort_values('mean')

print("\n各聚类的γ统计（越负越好）:")
print(cluster_stats)

# 识别最好的3个聚类
best_clusters = cluster_stats.head(3).index.tolist()
print(f"\n最优的3个聚类区域: {best_clusters}")

# 从最优区域选择分子
optimal_molecules = all_data[all_data['cluster'].isin(best_clusters)].copy()
optimal_molecules = optimal_molecules.sort_values('gamma')

print(f"\n最优区域中的Top 7分子:")
print(optimal_molecules[['Molecule_ID', 'gamma', 'cluster']].head(7))

# Step 8: 模拟实验验证
print("\n" + "="*60)
print("Step 8: 模拟实验验证")
print("="*60)

# 选择最好的2个分子进行"实验验证"
best_molecules = optimal_molecules.head(2)
print(f"\n选择进行实验验证的分子:")
for idx, mol in best_molecules.iterrows():
    print(f"- {mol['Molecule_ID']}: γ = {mol['gamma']:.3f}, Predict CE = {mol['CE']:.1f}%")

# 模拟实验结果
print("\n模拟实验结果:")
print("添加剂\t\t初始CE(%)\t极化电压(mV)\t500次循环后CE(%)")
print("-"*60)
print("纯ZnSO4\t\t71.4\t\t272\t\t96.4")
for idx, mol in best_molecules.iterrows():
    # 基于γ值生成模拟的实验数据
    init_ce = 85 + 10 * mol['gamma'] + np.random.normal(0, 2)
    voltage = 250 + 50 * mol['gamma'] + np.random.normal(0, 10)
    cycle_ce = 98 + 2 * mol['gamma'] + np.random.normal(0, 0.5)
    print(f"{mol['Molecule_ID']}\t\t{init_ce:.1f}\t\t{voltage:.0f}\t\t{cycle_ce:.1f}")

# 总结
print("\n" + "="*60)
print("完整流程总结")
print("="*60)
print("""
1. ✓ 生成了38个训练分子和27个测试分子的虚拟数据
2. ✓ 验证了表面自由能γ与库伦效率的二次关系 (R²={:.3f})
3. ✓ 使用类SISSO方法选择了{}个关键特征
4. ✓ 构建预测模型，交叉验证R²={:.3f}
5. ✓ 使用RTE降维和K-means聚类(K={})
6. ✓ 识别了{}个最优添加剂区域
7. ✓ 选择并"验证"了最优的添加剂分子
""".format(r2_score, len(top_features), np.mean(cv_scores), K, len(best_clusters)))

print("AI驱动的电解质添加剂筛选流程复现完成！")