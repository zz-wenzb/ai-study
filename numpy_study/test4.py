import numpy as np

tang_array = np.array([[1.5, 1.3, 7.5],
                       [5.6, 7.8, 1.2]])
# 排序
print(np.sort(tang_array))
print(np.sort(tang_array, axis=0))
# 排序索引
print(np.argsort(tang_array))
# 等差数列
print(np.linspace(0, 10, 10))
