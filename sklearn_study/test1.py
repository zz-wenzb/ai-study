from sklearn.datasets import load_iris

iris = load_iris()
# print(iris.keys())
X, y = iris.data, iris.target
print('X:', type(X))
print('X:', X[:5, :])
print('X:', X.shape)
# print('X:', X.shape[0])
# print('X:', X.shape[1])
print('y:', y)
print('y:', type(y))
print('y:', y.shape)
# print('y:', y.shape[0])
print(f"特征数量: {X.shape[1]}")
print(f"类别数量: {len(set(y))}")
