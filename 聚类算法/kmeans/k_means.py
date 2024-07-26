import numpy as np


class KMeans:
    """
    KMeans聚类算法实现类。

    Attributes:
        data: 输入的数据集。
        num_clusters: 需要划分的聚类数量。
    """

    def __init__(self, data, num_clusters):
        self.data = data
        self.num_clusters = num_clusters

    def train(self, max_iterations):
        """
        KMeans算法的训练方法。

        Parameters:
            max_iterations: 最大迭代次数。

        Returns:
            centroids: 最终的聚类中心点。
            closest_centroids_ids: 每个数据点所属的聚类中心索引。
        """
        # 初始化聚类中心
        # 1.先随机选择K个中心点
        centroids = KMeans.centroids_init(self.data, self.num_clusters)
        # 初始化每个数据点到聚类中心的最近距离索引
        # 2.开始训练
        num_examples = self.data.shape[0]
        closest_centroids_ids = np.empty((num_examples, 1))
        # 迭代更新聚类中心
        for _ in range(max_iterations):
            # 更新每个数据点所属的聚类中心索引
            # 3得到当前每一个样本点到K个中心点的距离，找到最近的
            closest_centroids_ids = KMeans.centroids_find_closest(self.data, centroids)
            # 根据数据点归属更新聚类中心
            # 4.进行中心点位置更新
            centroids = KMeans.centroids_compute(self.data, closest_centroids_ids, self.num_clusters)
        return centroids, closest_centroids_ids

    @staticmethod
    def centroids_init(data, num_clusters):
        """
        初始化聚类中心的方法。

        Parameters:
            data: 输入的数据集。
            num_clusters: 聚类数量。

        Returns:
            centroids: 随机选取的聚类中心。
        """
        # 从数据集中随机选取num_clusters个数据作为初始聚类中心
        num_examples = data.shape[0]
        random_ids = np.random.permutation(num_examples)
        centroids = data[random_ids[:num_clusters], :]
        return centroids

    @staticmethod
    def centroids_find_closest(data, centroids):
        """
        计算每个数据点距离最近的聚类中心的方法。

        Parameters:
            data: 输入的数据集。
            centroids: 当前的聚类中心。

        Returns:
            closest_centroids_ids: 每个数据点距离最近的聚类中心的索引。
        """
        num_examples = data.shape[0]
        num_centroids = centroids.shape[0]
        closest_centroids_ids = np.zeros((num_examples, 1))
        # 遍历每个数据点，计算其与所有聚类中心的距离
        for example_index in range(num_examples):
            distance = np.zeros((num_centroids, 1))
            for centroid_index in range(num_centroids):
                distance_diff = data[example_index, :] - centroids[centroid_index, :]
                distance[centroid_index] = np.sum(distance_diff ** 2)
            # 标记距离最近的聚类中心索引
            closest_centroids_ids[example_index] = np.argmin(distance)
        return closest_centroids_ids

    @staticmethod
    def centroids_compute(data, closest_centroids_ids, num_clusters):
        """
        更新聚类中心的方法。

        Parameters:
            data: 输入的数据集。
            closest_centroids_ids: 每个数据点所属的聚类中心索引。
            num_clusters: 聚类数量。

        Returns:
            centroids: 更新后的聚类中心。
        """
        num_features = data.shape[1]
        centroids = np.zeros((num_clusters, num_features))
        # 根据每个聚类中的数据点计算新的聚类中心
        for centroid_id in range(num_clusters):
            closest_ids = closest_centroids_ids == centroid_id
            centroids[centroid_id] = np.mean(data[closest_ids.flatten(), :], axis=0)
        return centroids
