from sklearn.datasets import load_digits
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 加载数据
digits = load_digits()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=42)

# 创建分类器
mnb = MultinomialNB()

# 训练模型
mnb.fit(X_train, y_train)

# 预测
y_pred = mnb.predict(X_test)

# 评估模型
print(classification_report(y_test, y_pred))

# 创建分类器
gnb = GaussianNB()

# 训练模型
gnb.fit(X_train, y_train)

# 预测
y_pred = gnb.predict(X_test)

# 评估模型
print(classification_report(y_test, y_pred))

# 创建分类器
bnb = BernoulliNB()

# 训练模型
bnb.fit(X_train, y_train)

# 预测
y_pred = bnb.predict(X_test)

# 评估模型
print(classification_report(y_test, y_pred))