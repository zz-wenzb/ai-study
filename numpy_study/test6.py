import numpy as np

education_years = np.array([10, 12, 12, 14, 14, 16, 16, 18, 20, 20])
# 数组重塑为一个列向量，其中-1表示自动推断行数，1表示列数为1。
education_years.reshape(-1, 1)
print(education_years)
