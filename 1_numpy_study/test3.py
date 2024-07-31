import numpy as np

tang_array = np.array([[1, 2, 3], [4, 5, 6]])
# 求和
print(np.sum(tang_array))
# 0 -2 y轴；1 -1 x轴.超出部分报错
print(np.sum(tang_array, axis=-2))

# 乘积
print(tang_array.prod())
print(tang_array.prod(axis=0))

print(tang_array.min())
print(tang_array.min(axis=1))

# 找到最小索引位置
print(tang_array.argmin())
print(tang_array.argmin(axis=1))
# 找到最大索引位置
print(tang_array.argmax())
print(tang_array.argmax(axis=1))

# 平均数
print(tang_array)
print(tang_array.mean())
print(tang_array.mean(axis=0))
print(tang_array.mean(axis=1))

# 标准差
print(tang_array.std())

# 方差
print(tang_array.var())

print(tang_array.clip(2, 4))

# 四舍五入
tang_array = np.array([1.2, 3.56, 6.41])
print(tang_array.round())
print(tang_array.round(decimals=1))
