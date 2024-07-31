from sklearn import datasets  # 引入数据集,sklearn包含众多数据集
from sklearn.model_selection import train_test_split  # 将数据分为测试集和训练集
from sklearn.neighbors import KNeighborsClassifier  # 利用邻近点方式训练数据

###引入数据###
iris = datasets.load_iris()  # 引入iris鸢尾花数据,iris数据包含4个特征变量
iris_X = iris.data  # 特征变量
iris_y = iris.target  # 目标值
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y,
                                                    test_size=0.3)  # 利用train_test_split进行将训练集和测试集进行分开，test_size占30%
print(y_train)  # 我们看到训练数据的特征值分为3类
'''
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
 '''

###训练数据###
# 该函数用于创建一个KNeighborsClassifier对象，该对象是一种用于分类任务的机器学习模型。它使用KNN（K-Nearest Neighbors）算法来预测新样本的类别。在创建对象时，
# 可以设置一些参数来调整模型的行为，如n_neighbors（用于决策的最近邻数目）等。
knn = KNeighborsClassifier()  # 引入训练方法
knn.fit(X_train, y_train)  # 进行填充测试数据进行训练

###预测数据###
print(knn.predict(X_test))  # 预测特征值
'''
[1 1 1 0 2 2 1 1 1 0 0 0 2 2 0 1 2 2 0 1 0 0 0 0 0 0 2 1 0 0 0 1 0 2 0 2 0
 1 2 1 0 0 1 0 2]
'''
print(y_test)  # 真实特征值
'''
[1 1 1 0 1 2 1 1 1 0 0 0 2 2 0 1 2 2 0 1 0 0 0 0 0 0 2 1 0 0 0 1 0 2 0 2 0
 1 2 1 0 0 1 0 2]
'''
