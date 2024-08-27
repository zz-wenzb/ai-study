from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import nltk
nltk.download('punkt')  # 下载NLTK的分词器数据
from nltk.tokenize import word_tokenize

# 示例文本
text_data = [
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",
    "The best time to plant a tree was twenty years ago.",
    "Success is not final, failure is not fatal: it is the courage to continue that counts."
]

# 预处理函数
def preprocess(text):
    return [token for token in simple_preprocess(text) if token not in STOPWORDS and len(token) > 3]

# 将文本数据预处理
processed_data = [preprocess(doc) for doc in text_data]

# 保存处理后的数据为 GloVe 输入格式
with open('glove_input.txt', 'w', encoding='utf-8') as file:
    for sentence in processed_data:
        file.write(' '.join(sentence) + '\n')

# 使用 GloVe 工具训练模型
glove_input_file = 'glove_input.txt'
glove_output_file = 'glove_output.txt'

# GloVe 命令行工具路径
glove_tool_path = '/path/to/glove/build/glove'

# 执行 GloVe 命令
# !{glove_tool_path} -input-file {glove_input_file} -output-file {glove_output_file} \
#                     -x-max 10 -iter 100 -vector-size 100 -binary 0 -threads 4 -min-count 5

# 将 GloVe 输出转换为 Gensim 可读格式
glove2word2vec(glove_output_file, 'glove_word2vec.txt')

# 加载转换后的模型
glove_w2v_model = KeyedVectors.load_word2vec_format('glove_word2vec.txt')

# 获取词向量
word_vector = glove_w2v_model["success"]
print(word_vector)

# 计算相似度
similarity = glove_w2v_model.similarity("success", "failure")
print(f"Similarity between 'success' and 'failure': {similarity}")

# 类比推理
result = glove_w2v_model.most_similar(positive=["journey"], negative=["thousand"])
print(result)