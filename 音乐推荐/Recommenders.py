import numpy as np
import pandas


class popularity_recommender_py():
    """
    基于流行度的推荐系统类。
    """

    def __init__(self):
        """
        初始化推荐系统实例。
        """
        self.train_data = None  # 训练数据
        self.user_id = None  # 用户ID字段名
        self.item_id = None  # 项目ID字段名
        self.popularity_recommendations = None  # 流行度推荐结果

    def create(self, train_data, user_id, item_id):
        """
        创建推荐系统。

        参数:
        train_data: pandas.DataFrame, 训练数据集。
        user_id: str, 用户ID字段名。
        item_id: str, 项目ID字段名。
        """
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

        # 按项目分组，计算每个项目的用户数量，重置索引
        train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()
        # 重命名列名为'score'
        train_data_grouped.rename(columns={user_id: 'score'}, inplace=True)

        # 按'score'和项目ID排序
        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending=[0, 1])

        # 添加排名列
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')

        # 取排名前10的项目作为推荐结果
        self.popularity_recommendations = train_data_sort.head(10)

    def recommend(self, user_id):
        """
        为用户推荐流行项目。

        参数:
        user_id: str, 用户ID。

        返回:
        pandas.DataFrame, 包含推荐结果的DataFrame。
        """
        user_recommendations = self.popularity_recommendations

        # 添加用户ID列
        user_recommendations['user_id'] = user_id

        # 调整列的顺序
        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]

        return user_recommendations


class item_similarity_recommender_py():
    """
    基于物品相似度的推荐系统类。
    """

    def __init__(self):
        """
        初始化推荐系统实例。
        """
        self.train_data = None  # 训练数据
        self.user_id = None  # 用户ID字段名
        self.item_id = None  # 项目ID字段名
        self.cooccurence_matrix = None  # 共现矩阵
        self.songs_dict = None  # 歌曲字典
        self.rev_songs_dict = None  # 逆向歌曲字典
        self.item_similarity_recommendations = None  # 物品相似度推荐结果

    def get_user_items(self, user):
        """
        获取用户听过的歌曲。

        参数:
        user: str, 用户ID。

        返回:
        list, 用户听过的歌曲列表。
        """
        user_data = self.train_data[self.train_data[self.user_id] == user]
        user_items = list(user_data[self.item_id].unique())

        return user_items

    def get_item_users(self, item):
        """
        获取听过某首歌曲的所有用户。

        参数:
        item: str, 歌曲ID。

        返回:
        set, 听过该歌曲的用户集合。
        """
        item_data = self.train_data[self.train_data[self.item_id] == item]
        item_users = set(item_data[self.user_id].unique())

        return item_users

    def get_all_items_train_data(self):
        """
        获取训练数据集中的所有歌曲。

        返回:
        list, 训练数据集中的所有歌曲列表。
        """
        all_items = list(self.train_data[self.item_id].unique())

        return all_items

    def construct_cooccurence_matrix(self, user_songs, all_songs):
        """
        构建共现矩阵。

        参数:
        user_songs: list, 用户听过的歌曲列表。
        all_songs: list, 所有歌曲列表。

        返回:
        np.matrix, 共现矩阵。
        """
        user_songs_users = []
        for i in range(0, len(user_songs)):
            user_songs_users.append(self.get_item_users(user_songs[i]))

        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))), float)

        for i in range(0, len(all_songs)):
            songs_i_data = self.train_data[self.train_data[self.item_id] == all_songs[i]]
            users_i = set(songs_i_data[self.user_id].unique())

            for j in range(0, len(user_songs)):
                users_j = user_songs_users[j]
                users_intersection = users_i.intersection(users_j)
                if len(users_intersection) != 0:
                    users_union = users_i.union(users_j)
                    cooccurence_matrix[j, i] = float(len(users_intersection)) / float(len(users_union))
                else:
                    cooccurence_matrix[j, i] = 0

        return cooccurence_matrix

    def generate_top_recommendations(self, user, cooccurence_matrix, all_songs, user_songs):
        """
        生成顶级推荐结果。

        参数:
        user: str, 用户ID。
        cooccurence_matrix: np.matrix, 共现矩阵。
        all_songs: list, 所有歌曲列表。
        user_songs: list, 用户听过的歌曲列表。

        返回:
        pandas.DataFrame, 包含推荐结果的DataFrame。
        """
        print("Non zero values in cooccurence_matrix :%d" % np.count_nonzero(cooccurence_matrix))

        user_sim_scores = cooccurence_matrix.sum(axis=0) / float(cooccurence_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()

        sort_index = sorted(((e, i) for i, e in enumerate(list(user_sim_scores))), reverse=True)

        columns = ['user_id', 'song', 'score', 'rank']
        df = pandas.DataFrame(columns=columns)

        rank = 1
        for i in range(0, len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <= 10:
                df.loc[len(df)] = [user, all_songs[sort_index[i][1]], sort_index[i][0], rank]
                rank = rank + 1

        if df.shape[0] == 0:
            print("The current user has no songs for training the item similarity based recommendation model.")
            return -1
        else:
            return df

    def create(self, train_data, user_id, item_id):
        """
        创建推荐系统。

        参数:
        train_data: pandas.DataFrame, 训练数据集。
        user_id: str, 用户ID字段名。
        item_id: str, 项目ID字段名。
        """
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

    def recommend(self, user):
        """
        为用户生成推荐结果。

        参数:
        user: str, 用户ID。

        返回:
        pandas.DataFrame, 包含推荐结果的DataFrame。
        """
        user_songs = self.get_user_items(user)
        print("No. of unique songs for the user: %d" % len(user_songs))
        all_songs = self.get_all_items_train_data()
        print("no. of unique songs in the training set: %d" % len(all_songs))
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)
        return df_recommendations

    def get_similar_items(self, item_list):
        """
        获取与给定列表中的项目相似的项目。

        参数:
        item_list: list, 项目ID列表。

        返回:
        pandas.DataFrame, 包含相似项目推荐结果的DataFrame。
        """
        user_songs = item_list
        all_songs = self.get_all_items_train_data()
        print("no. of unique songs in the training set: %d" % len(all_songs))
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)
        user = ""
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)
        return df_recommendations
