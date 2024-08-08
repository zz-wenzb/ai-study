import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from matplotlib import rcParams

# 设置matplotlib的字体
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 简单线性回归
education_years = np.array([10, 12, 14, 16, 18, 20])  # 教育年数
income = np.array([26, 30, 38, 50, 58, 60])  # 个人年收入（以千美元计）

simple_model = LinearRegression()
simple_model.fit(education_years.reshape(-1, 1), income)

# 预测
income_pred = simple_model.predict(education_years.reshape(-1, 1))
print(income_pred)
# 计算评估指标
simple_mse = mean_squared_error(income, income_pred)

# 绘制结果
plt.figure(figsize=(10, 5))
plt.scatter(education_years, income, color='blue', label='实际收入')
plt.plot(education_years, income_pred, color='red', label='预测收入')
plt.title('教育年数与个人收入关系')
plt.xlabel('教育年数')
plt.ylabel('年收入（千美元）')
plt.legend()
plt.show()

print(f"简单线性回归评估指标:\n均方误差(MSE): {simple_mse:.2f}")

# 分析结果
analysis_text = print(f"""
简单线性回归结果分析：
每增加一年的教育，预期收入增加约 {simple_model.coef_[0]:.2f} 千美元。
""")
