from sklearn.datasets import load_breast_cancer

# 乳腺癌数据集
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
print('X:', type(X))
print('X:', X.shape)
# print('X:', X.shape[0])
# print('X:', X.shape[1])
print('y:', type(y))
print('y:', y.shape)
# print('y:', y.shape[0])
print(f"特征数量: {X.shape[1]}")
print(f"样本数量: {len(set(y))}")
print(f"样本数量: {len(y)}，其中0代表良性，1代表恶性")
