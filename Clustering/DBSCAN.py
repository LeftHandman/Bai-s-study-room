import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

# 生成示例数据
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 创建DBSCAN模型
# 设置epsilon（邻域半径）和min_samples（一个聚类中的最小点数）
dbscan = DBSCAN(eps=0.5, min_samples=5)

# 拟合模型并进行聚类
labels = dbscan.fit_predict(X)

# 获取聚类结果
unique_labels = np.unique(labels)
num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # -1 表示噪声点

print(f"Number of clusters: {num_clusters}")
print(f"Cluster labels: {unique_labels}")

# 绘制聚类结果
plt.figure(figsize=(8, 6))
for label in unique_labels:
    if label == -1:
        # 黑色表示噪声点
        plt.scatter(X[labels == label, 0], X[labels == label, 1], color='black', s=50, label='Noise')
    else:
        # 为每个聚类分配不同的颜色
        plt.scatter(X[labels == label, 0], X[labels == label, 1], s=50, label=f'Cluster {label}')

plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
