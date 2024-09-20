import numpy as np
import pandas as pd
from prophet import Prophet
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('data/BABA.csv')
# print(df.head())

# 筛选特定列
new_df = df[['Date', 'Close']]
# 为Prophet Model 时间序列:ds, price:y重命名列
new_df.columns = ['ds', 'y']
# print(new_df)

# 初始化模型
model = Prophet()
# 训练模型
model.fit(new_df)
future = model.make_future_dataframe(periods=360, freq='D')  # 预测未来360天
# 模型预测
guess = model.predict(future)
print(guess)

model.plot(guess).show()
