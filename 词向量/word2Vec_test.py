from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import nltk

# 下载NLTK的分词器数据
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# 示例文本
text_data = [
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",
    "The best time to plant a tree was twenty years ago.",
    "Success is not final, failure is not fatal: it is the courage to continue that counts."
]


# 预处理函数
# 去除停用词和长度小于等于3的词
def preprocess(text):
    return [token for token in simple_preprocess(text) if token not in STOPWORDS and len(token) > 3]


# 将文本数据预处理
processed_data = [preprocess(doc) for doc in text_data]

# 创建Word2Vec模型
# 使用 Skip-gram 模型，词向量维度为100，上下文窗口大小为5，忽略词频少于1的词，使用4个线程训练
model = Word2Vec(
    sentences=processed_data,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1
)

# 训练模型
model.train(processed_data, total_examples=len(processed_data), epochs=100)

# 保存模型
model.save("word2vec.model")

# 加载模型
# model = Word2Vec.load("word2vec.model")

# 获取词向量
word_vector = model.wv["success"]
print(word_vector)

# 计算相似度
similarity = model.wv.similarity("success", "failure")
print(f"Similarity between 'success' and 'failure': {similarity}")

# 类比推理
result = model.wv.most_similar(positive=["journey"], negative=["thousand"])
print(result)
