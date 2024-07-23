import numpy as np

array = np.array([1, 2, 3, 4, 5])
print(array)
print(type(array))
# 访问数组的第一个和最后一个元素
print(array[0])
print(array[4])
print(array[-1])
# 打印数组的形状，对于一维数组，形状为一个元素的元组。
print(array.shape)
# 一维数组的切片
print(array[:3])

array2 = array + 1
print(array2)

array3 = array2 + array
print(array3)

array4 = array2 * array
print(array4)

a = np.array([[1, 2, 3], [4, 5, 6]])
print(a)
print(a.shape)
