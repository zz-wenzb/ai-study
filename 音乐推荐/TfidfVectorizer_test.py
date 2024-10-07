from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    stop_words='english',  # 可选参数，去除停用词
    max_df=0.5,  # 忽略高于这个值的最高文档频率的项（例如，术语“is”在大多数文档中都会出现）
    min_df=1,  # 忽略低于这个值的最低文档频率的项
    use_idf=True,  # 启用逆文档频率特征
    smooth_idf=True,  # 平滑 IDF 值，防止分母为零
    sublinear_tf=True  # 应用子线性函数于词频
)

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out())  # 输出所有特征名称
print(X.toarray())  # 输出转换后的矩阵
