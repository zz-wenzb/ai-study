import pandas as pd

data = {
    'Area': [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],  # 面积（平方米）
    'Bedrooms': [1, 1, 2, 2, 3, 3, 3, 4, 4, 4],  # 卧室数量
    'Age': [10, 5, 5, 10, 1, 1, 5, 10, 5, 1],  # 房龄
    'Distance': [5, 6, 5, 6, 4, 3, 4, 2, 1, 2],  # 距离市中心的距离（公里）
    'Price': [292, 316, 332, 355, 374, 396, 410, 438, 451, 466]  # 房价（以千美元计），添加噪声
}

df = pd.DataFrame(data)
# print(df)
# print(df[['Area', 'Bedrooms', 'Age', 'Distance']])
# print(df[['Price']])
# print(df['Price'])

print(type(df))
