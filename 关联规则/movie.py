# 分析MovieLens 电影分类中的频繁项集和关联规则
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# 数据加载
movies = pd.read_csv('ml-latest-small/movies.csv')
# 将genres进行one-hot编码（离散特征有多少取值，就用多少维来表示这个特征）
genres = movies.drop(labels='genres', axis=1)
movies_hot_encoded = genres.join(movies.genres.str.get_dummies(sep='|'))
pd.options.display.max_columns = 100
# print(movies_hot_encoded.head())

# 将movieId, title设置为index
movies_hot_encoded.set_index(['movieId', 'title'], inplace=True)
# 挖掘频繁项集，最小支持度为0.02
item_sets = apriori(movies_hot_encoded, use_colnames=True, min_support=0.02)
# # 按照支持度从大到小
item_sets = item_sets.sort_values(by="support", ascending=False)
# print(item_sets)
# 根据频繁项集计算关联规则，设置最小提升度为2
rules = association_rules(item_sets, metric='lift', min_threshold=2)
# 按照提升度从大到小进行排序
rules = rules.sort_values(by="lift", ascending=False)
rules.to_csv('./rules.csv')
print(rules)
