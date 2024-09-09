import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pydot
from sklearn.tree import export_graphviz

pd.set_option('display.max_columns', None)
features = pd.read_csv('data/temps.csv')
# print(features.head(10))
#    year  month  day   week  temp_2  temp_1  average  actual  friend
# 0  2016      1    1    Fri      45      45     45.6      45      29
# print(features.shape)
# (348, 9)
# print(features.describe())

years = features['year']
months = features['month']
days = features['day']
dates = [(datetime.date(year, month, day).strftime('%Y-%m-%d')) for year, month, day in zip(years, months, days)]
# print(dates[:10])

plt.style.use('fivethirtyeight')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
# fig.autofmt_xdate(rotation=45)

ax1.plot(dates, features['actual'])
ax1.set_title('actual')
ax1.set_xlabel('')
ax1.set_ylabel('temperature')

ax2.plot(dates, features['temp_1'])
ax2.set_title('temp_1')
ax2.set_xlabel('')
ax2.set_ylabel('temperature')

ax3.plot(dates, features['temp_2'])
ax3.set_title('temp_2')
ax3.set_xlabel('')
ax3.set_ylabel('temperature')

ax4.plot(dates, features['friend'])
ax4.set_title('friend')
ax4.set_xlabel('')
ax4.set_ylabel('temperature')

plt.tight_layout(pad=2)
plt.show()

# 数据预处理
# 独热编码
features = pd.get_dummies(features)
# print(features.head())
# print(features.shape)
labels = np.array(features['actual'])
# print(type(labels))
# print(labels)
# print(features.columns)
features = features.drop('actual', axis=1)

# print(features.head())
# print(features.columns)
feature_list = list(features.columns)
# print(feature_list)
features = np.array(features)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                            random_state=42)
# print('训练集特征:', train_features.shape)
# print('训练集标签:', train_labels.shape)
# print('测试集特征:', test_features.shape)
# print('测试集标签:', test_labels.shape)

rf = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
rf.fit(train_features, train_labels)

predictions = rf.predict(test_features)

errors = abs(predictions - test_labels)

mape = 100 * (errors / test_labels)
# mape越低，模型泛化越好
print('MAPE:', np.mean(mape))

# 树展示
tree = rf.estimators_[5]

# # 导出成dot文件
# export_graphviz(tree, out_file='tree.dot', feature_names=feature_list, rounded=True, precision=1)
#
# # 绘图
# (graph,) = pydot.graph_from_dot_file('tree.dot')
#
# # 展示
# graph.write_png('tree.png')


# 得到特征重要性
importances = list(rf.feature_importances_)

# 转换格式
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# print(feature_importances)
# print(type(feature_importances))
# 排序
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

# 对应进行打印
# [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

# 选择最重要的那两个特征来试一试
rf_most_important = RandomForestRegressor(n_estimators=1000, random_state=42)

# 拿到这俩特征
important_indices = [feature_list.index('temp_1'), feature_list.index('average')]
train_important = train_features[:, important_indices]
test_important = test_features[:, important_indices]

# 重新训练模型
rf_most_important.fit(train_important, train_labels)

# 预测结果
predictions = rf_most_important.predict(test_important)

errors = abs(predictions - test_labels)

# 评估结果

mape = np.mean(100 * (errors / test_labels))

print('mape:', mape)

# 转换成list格式
x_values = list(range(len(importances)))
print(x_values)

# 绘图
plt.bar(x_values, importances, orientation='vertical')

# x轴名字
plt.xticks(x_values, feature_list, rotation='vertical')

# 图名
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.title('Variable Importances')
plt.tight_layout()
plt.show()

# 日期数据
months = features[:, feature_list.index('month')]
days = features[:, feature_list.index('day')]
years = features[:, feature_list.index('year')]

# 转换日期格式
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

# 创建一个表格来存日期和其对应的标签数值
true_data = pd.DataFrame(data={'date': dates, 'actual': labels})

# 同理，再创建一个来存日期和其对应的模型预测值
months = test_features[:, feature_list.index('month')]
days = test_features[:, feature_list.index('day')]
years = test_features[:, feature_list.index('year')]

test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in
              zip(years, months, days)]

test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]

predictions_data = pd.DataFrame(data={'date': test_dates, 'prediction': predictions})

# 日期数据
months = features[:, feature_list.index('month')]
days = features[:, feature_list.index('day')]
years = features[:, feature_list.index('year')]

# 转换日期格式
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

# 创建一个表格来存日期和其对应的标签数值
true_data = pd.DataFrame(data={'date': dates, 'actual': labels})

# 同理，再创建一个来存日期和其对应的模型预测值
months = test_features[:, feature_list.index('month')]
days = test_features[:, feature_list.index('day')]
years = test_features[:, feature_list.index('year')]

test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in
              zip(years, months, days)]

test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]

predictions_data = pd.DataFrame(data={'date': test_dates, 'prediction': predictions})
# 真实值
plt.plot(true_data['date'], true_data['actual'], 'b-', label='actual')

# 预测值
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label='prediction')
plt.xticks(rotation='vertical')  # vertical horizontal
plt.legend()

# 图名
plt.xlabel('Date')
plt.ylabel('Maximum Temperature (F)')
plt.title('Actual and Predicted Values')
plt.tight_layout()
plt.show()
