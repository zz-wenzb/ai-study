import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

# 加载数据集
iris = load_iris()
X, y = iris.data, iris.target

# 创建并训练模型
model = LogisticRegression(max_iter=1000)

# 设置自助法重复次数
n_iterations = 1000
bootstrapped_scores = []

for i in range(n_iterations):
    # 有放回地随机抽样生成新的训练集
    X_resampled, y_resampled = resample(X, y)

    # 在新的样本上训练模型
    model.fit(X_resampled, y_resampled)

    # 计算准确率
    score = model.score(X_resampled, y_resampled)
    bootstrapped_scores.append(score)

# 输出每次自助法的准确率
print("Bootstrapped accuracy scores:", bootstrapped_scores)

# 计算平均准确率和置信区间
mean_accuracy = np.mean(bootstrapped_scores)
confidence_interval = np.percentile(bootstrapped_scores, [2.5, 97.5])
print("Mean accuracy:", mean_accuracy)
print("Confidence interval:", confidence_interval)
