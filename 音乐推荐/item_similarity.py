import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# 示例数据
data = {
    'UserID': [1, 1, 1, 2, 2, 3, 3, 4, 4, 4],
    'ProductID': [1, 2, 3, 1, 2, 1, 2, 1, 2, 3],
    'Rating': [5, 3, 4, 4, 2, 3, 1, 4, 5, 3]
}

# 转换为 DataFrame
df = pd.DataFrame(data)

# 创建用户-商品的评分矩阵
pivot_table = df.pivot(index='ProductID', columns='UserID', values='Rating').fillna(0)
pivot_matrix = pivot_table.values

# 转换为稀疏矩阵
sparse_matrix = csr_matrix(pivot_matrix)

# 计算商品间的余弦相似度
product_similarity = cosine_similarity(sparse_matrix)

# 获取用户已评分的商品
user_rated_products = df[df['UserID'] == 1]['ProductID'].tolist()

# 推荐未评分的商品
all_products = pivot_table.index.tolist()
unrated_products = list(set(all_products) - set(user_rated_products))

# 为 unrated_products 中的每一个商品计算相似度得分
recommendations = []
for product in unrated_products:
    similarity_scores = [(similarity, product) for product_id, similarity in enumerate(product_similarity[product])]
    similarity_scores = sorted(similarity_scores, key=lambda x: x[0], reverse=True)
    recommendations.extend(similarity_scores[:10])  # 选取前 10 个最相似的商品

# 排序并输出推荐的商品
recommendations = sorted(recommendations, key=lambda x: x[0], reverse=True)
recommended_products = [product for similarity, product in recommendations if product not in user_rated_products][:10]
print("Recommended products:", recommended_products)
