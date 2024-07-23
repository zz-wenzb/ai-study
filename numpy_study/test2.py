import numpy as np

print("一维数组")
tang_list = [1, 2, 3, 4, 5]
tang_array = np.array(tang_list)
# 数组类型
print(tang_array.dtype)
# 数组元素大小
print(tang_array.itemsize)
# 数组形状
print(tang_array.shape)
# 数组大小
print(tang_array.size)
print(np.size(tang_array))
# 数组维数
print(tang_array.ndim)
# 数组填充
# tang_array.fill(0)
# print(tang_array)

# 数组切片
print(tang_array[1:3])
print(tang_array[-2:])

print("二维数组")
tang_array = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
print(tang_array.shape)
print(tang_array.ndim)
# 二维数组切片
print(tang_array[1, 1])
print(tang_array[1])

tang_array2 = tang_array
print(tang_array2)
# 二维数组赋值
tang_array2[1, 1] = 100
print(tang_array)
print(tang_array2)
# 二维数组复制
tang_array3 = tang_array.copy()
print(tang_array3)
tang_array3[1, 1] = 10
print(tang_array)
print(tang_array3)

print("arange 数组")
a = np.arange(0, 100, 10)
print(a)

b = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=bool)
print(b)
print(a[b])

random_array = np.random.rand(10)
print(random_array)

mask = random_array > 0.5
print(mask)
print(random_array[mask])

tang_array = np.array([10, 20, 30, 40, 50])
print(tang_array[np.where(tang_array > 30)])

tang_array = np.array([1, 10, 3.5, 'str'])

print(tang_array.dtype)
# print(tang_array * 2)
