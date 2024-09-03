import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 生成示例数据
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 创建KMeans模型
# 需要指定聚类的数量k
kmeans = KMeans(n_clusters=4, random_state=0)

# 拟合模型并进行聚类
labels = kmeans.fit_predict(X)

# 获取聚类中心
centers = kmeans.cluster_centers_

# 绘制聚类结果
plt.figure(figsize=(8, 6))
for label in np.unique(labels):
    plt.scatter(X[labels == label, 0], X[labels == label, 1], s=50, label=f'Cluster {label}')

# 绘制聚类中心
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroids')

plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
