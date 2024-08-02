from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
labels = ['red', 'blue', 'red', 'green', 'blue', 'yellow']
le.fit(labels)
encoded_labels = le.transform(labels)
print(encoded_labels)  # 输出可能是[0, 1, 0, 2, 1, 3]
decoded_labels = le.inverse_transform([0, 1, 2, 1, 3])
print(decoded_labels)  # 输出可能是['red', 'blue', 'green', 'blue', 'yellow']
