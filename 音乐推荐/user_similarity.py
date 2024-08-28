import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 示例数据集
ratings_data = {
    'UserID': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5],
    'ItemID': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
    'Rating': [5, 3, 2, 4, 3, 1, 3, 2, 4, 2, 4, 5, 1]
}

# 创建 DataFrame
ratings_df = pd.DataFrame(ratings_data)

# 构建用户-物品评分矩阵
user_item_ratings = ratings_df.pivot(index='UserID', columns='ItemID', values='Rating').fillna(0)

# 计算用户间相似度
user_similarity = cosine_similarity(user_item_ratings)


# 定义一个函数来获取推荐物品
def get_recommendations(user_id, num_recommendations=3):
    # 获取该用户已评分的物品
    rated_items = ratings_df[ratings_df['UserID'] == user_id]['ItemID']

    # 找到与目标用户最相似的用户
    similar_users = np.argsort(user_similarity[user_id - 1])[::-1][1:num_recommendations + 1]

    # 获取这些相似用户的评分记录
    similar_user_ratings = user_item_ratings.iloc[similar_users]

    # 计算未评分物品的预测评分
    unrated_items = user_item_ratings.drop(rated_items, errors='ignore').columns
    unrated_item_ratings = similar_user_ratings[unrated_items].mean(axis=0)

    # 获取最高评分的未评分物品
    top_unrated_items = unrated_items[np.argsort(unrated_item_ratings)[::-1]].tolist()

    # 返回推荐物品
    return top_unrated_items[:num_recommendations]


# 为用户 5 推荐物品
recommendations_for_user_1 = get_recommendations(5)
print("Recommendations for User 1:", recommendations_for_user_1)
