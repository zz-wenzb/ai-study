import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 加载数据
data = load_iris()
X, y = data.data, data.target

# 定义基模型
base_models = [
    ('Logistic Regression', LogisticRegression()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Support Vector Machine', SVC(probability=True))
]

# 定义元模型
meta_model = RandomForestClassifier()

# 初始化变量
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds)
S_train = np.zeros((X.shape[0], len(base_models)))
S_test = np.zeros((X.shape[0], len(base_models)))

# 创建堆叠层
for fold_id, (train_index, holdout_index) in enumerate(skf.split(X, y)):
    # 分割数据
    X_train, y_train = X[train_index], y[train_index]
    X_holdout, y_holdout = X[holdout_index], y[holdout_index]

    # 训练基模型并预测
    for i, (name, model) in enumerate(base_models):
        print(f"Training {name} on fold {fold_id + 1}...")
        model.fit(X_train, y_train)
        S_train[holdout_index, i] = model.predict(X_holdout)

# 训练元模型
meta_model.fit(S_train, y)

# 预测并评估
y_pred = meta_model.predict(S_train)
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy of the Stacking ensemble is {accuracy:.4f}")