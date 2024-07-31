from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 该函数用于加载一个数字数据集，其中包含了1797个样本，每个样本是一个8x8的图像，
# 代表了一个手写数字。函数返回一个包含样本数据和对应标签的数据集。
digits = load_digits()

# 分割数据集为训练集和测试集
# test_size表示测试集的比例，random_state表示随机数种子
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# 初始化支持向量机分类器
# 选择SVC作为模型是因为支持向量机在处理分类问题时，尤其是非线性分类问题上表现强大
# 这里使用gamma参数为0.001，是为了控制核函数的影响力，小的gamma值意味着模型更倾向于泛化，而不是过度拟合训练数据
clf = SVC(gamma=0.001)

# 在训练集上训练分类器
clf.fit(X_train, y_train)

# 使用训练好的分类器对测试集进行预测
y_pred = clf.predict(X_test)

# 计算预测的准确率
# accuracy_score返回预测正确的样本占总样本的比例
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 可视化测试集中的一些图像及其预测结果
test_images = X_test[:16]  # 取测试集的前16张图像
test_labels = y_test[:16]  # 和对应的标签
predictions = y_pred[:16]  # 和对应的预测结果

images_to_show = test_images.reshape(-1, 8, 8)  # 将图像数据重塑为8x8的形状

fig, axes = plt.subplots(4, 4)
fig.subplots_adjust(hspace=1, wspace=0.5)

for i, ax in enumerate(axes.flat):
    ax.imshow(images_to_show[i], cmap='gray_r')
    ax.set_title(f"True: {test_labels[i]}")
    ax.set_xlabel(f"Prediction: {predictions[i]}")
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()
