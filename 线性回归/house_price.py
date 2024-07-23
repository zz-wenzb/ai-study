import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import rcParams
from sklearn.model_selection import train_test_split

# 设置matplotlib的字体
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 简单线性回归
education_years = np.array([10, 12, 12, 14, 14, 16, 16, 18, 20, 20])  # 教育年数
income = np.array([26, 30, 30, 38, 40, 50, 52, 58, 60, 62])  # 个人年收入（以千美元计）

simple_model = LinearRegression()
simple_model.fit(education_years.reshape(-1, 1), income)

# 预测
income_pred = simple_model.predict(education_years.reshape(-1, 1))

# 计算评估指标
simple_mse = mean_squared_error(income, income_pred)
# 该函数用于计算收入预测值与实际收入之间的决定系数R^2，
# 用于评估模型的拟合优度。R^2值越接近1，表示模型对数据的拟合程度越好，越接近0表示模型对数据的拟合程度越差。
simple_r2 = r2_score(income, income_pred)

# 绘制结果
plt.figure(figsize=(10, 5))
plt.scatter(education_years, income, color='blue', label='实际收入')
plt.plot(education_years, income_pred, color='red', label='预测收入')
plt.title('教育年数与个人收入关系')
plt.xlabel('教育年数')
plt.ylabel('年收入（千美元）')
plt.legend()
plt.show()

# 多元线性回归
data = {
    'Area': [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],  # 面积（平方米）
    'Bedrooms': [1, 1, 2, 2, 3, 3, 3, 4, 4, 4],  # 卧室数量
    'Age': [10, 5, 5, 10, 1, 1, 5, 10, 5, 1],  # 房龄
    'Distance': [5, 6, 5, 6, 4, 3, 4, 2, 1, 2],  # 距离市中心的距离（公里）
    'Price': [292, 316, 332, 355, 374, 396, 410, 438, 451, 466]  # 房价（以千美元计），添加噪声
}
df = pd.DataFrame(data)

multi_model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(df[['Area', 'Bedrooms', 'Age', 'Distance']], df['Price'],
                                                    test_size=0.3, random_state=0)

multi_model.fit(X_train, y_train)

price_pred = multi_model.predict(X_test)
print(y_test)
print(price_pred)
# 计算评估指标
multi_mse = mean_squared_error(y_test, price_pred)
multi_r2 = r2_score(y_test, price_pred)

# 使用之前定义的数据和预测结果绘制图表
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='实际房价')
plt.plot(range(len(y_test)), price_pred, color='red', label='预测房价')
plt.title('房价预测比较')
plt.xlabel('样本编号')
plt.ylabel('房价（千美元）')
plt.legend()
plt.grid(True)
plt.show()

print(f"简单线性回归评估指标:\n均方误差(MSE): {simple_mse:.2f}\nR²分数: {simple_r2:.2f}")
print(f"多元线性回归评估指标:\n均方误差(MSE): {multi_mse:.2f}\nR²分数: {multi_r2:.2f}")

# 分析结果
analysis_text = print(f"""
简单线性回归结果分析：
每增加一年的教育，预期收入增加约 {simple_model.coef_[0]:.2f} 千美元。
预测模型的R²值为 {simple_r2:.2f}，表明模型拟合的质量较好。

多元线性回归结果分析：
根据模型，房屋面积、卧室数量、房龄和距市中心距离对房价的影响显著。
房价与面积的系数为 {multi_model.coef_[0]:.2f}，意味着面积每增加1平方米，房价平均增加 {multi_model.coef_[0]:.2f} 千美
""")
