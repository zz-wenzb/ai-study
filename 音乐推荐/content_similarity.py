import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 示例数据
movies_data = {
    'MovieID': [1, 2, 3, 4, 5],
    'Title': ['The Dark Knight', 'Inception', 'Interstellar', 'Pulp Fiction', 'The Godfather'],
    'Director': ['Christopher Nolan', 'Christopher Nolan', 'Christopher Nolan', 'Quentin Tarantino',
                 'Francis Ford Coppola'],
    'Actors': ['Christian Bale, Heath Ledger', 'Leonardo DiCaprio, Joseph Gordon-Levitt',
               'Matthew McConaughey, Anne Hathaway', 'John Travolta, Samuel L. Jackson', 'Marlon Brando, Al Pacino'],
    'Genres': ['Action, Crime, Drama', 'Action, Adventure, Sci-Fi', 'Adventure, Drama, Sci-Fi', 'Crime, Drama',
               'Crime, Drama']
}

ratings_data = {
    'UserID': [1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5],
    'MovieID': [1, 2, 1, 3, 2, 4, 1, 3, 5, 2, 5],
    'Rating': [5, 4, 4, 3, 3, 4, 4, 5, 3, 4, 3]
}

# 创建 DataFrame
movies_df = pd.DataFrame(movies_data)
ratings_df = pd.DataFrame(ratings_data)

# 构建用户-电影评分矩阵
user_movie_ratings = ratings_df.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0)

# 构建电影特征矩阵
# 将每行所有字段用空格连接
movie_features = movies_df[['Title', 'Director', 'Actors', 'Genres']].apply(lambda x: ' '.join(x), axis=1)
# 使用 TF-IDF 提取特征向量
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movie_features)

# 计算用户偏好向量
user_preferences = user_movie_ratings.apply(lambda x: (tfidf_matrix.T * x).sum(), axis=1)

# 计算相似度
user_preferences_matrix = user_preferences.values
similarity_matrix = cosine_similarity(user_preferences_matrix)


# 为每个用户推荐电影
def recommend_movies(user_id, n_recommendations=3):
    user_index = user_movie_ratings.index.get_loc(user_id)
    similarities = list(enumerate(similarity_matrix[user_index]))
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in similarities if i[0] != user_index][:n_recommendations]

    recommended_movies = movies_df.loc[movie_indices, 'Title'].tolist()
    return recommended_movies


# 为用户 1 推荐电影
recommended_movies = recommend_movies(1)
print("Recommended Movies for User 1:", recommended_movies)
