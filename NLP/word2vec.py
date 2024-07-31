from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import sys
import argparse
import random
from tempfile import gettempdir
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector

# Give a folder path as an argument with '--log_dir' to save
# TensorBoard summaries. Default is a log folder in current directory.
# 获取当前脚本所在目录路径，作为日志目录的默认位置
current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

# 初始化参数解析器
parser = argparse.ArgumentParser()
# 添加--log_dir参数，其默认值为在当前脚本目录下的"log"子目录
# 此目录用于存储TensorBoard摘要日志
parser.add_argument(
    '--log_dir',
    type=str,
    default=os.path.join(current_path, 'log'),
    help='用于TensorBoard日志的目录。')
# 解析命令行参数，返回任何未识别的参数供后续处理
FLAGS, unparsed = parser.parse_known_args()

# 如果日志目录不存在，则创建它
# 这确保了TensorBoard日志目录在继续之前已经准备就绪
# 如果TensorBoard变量目录不存在，则创建之。
if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)

# 定义数据下载的URL
# 第一步：下载数据。
url = 'http://mattmahoney.net/dc/'


# pylint: disable=redefined-outer-name
def maybe_download(filename, expected_bytes):
    """
    如果文件不存在，则从指定URL下载文件到临时目录，并验证文件大小是否符合预期。

    参数:
    filename: 需要下载的文件名。
    expected_bytes: 预期文件大小（字节）。

    返回:
    下载后文件的本地路径。
    """
    # 构建文件的本地路径
    local_filename = os.path.join(gettempdir(), filename)
    # 检查文件是否已存在于本地
    if not os.path.exists(local_filename):
        # 如果文件不存在，则从URL下载文件到指定的本地路径
        local_filename, _ = urllib.request.urlretrieve(url + filename, local_filename)
    # 获取本地文件的统计信息，包括文件大小等
    statinfo = os.stat(local_filename)
    # 检查本地文件大小是否与预期大小相符
    if statinfo.st_size == expected_bytes:
        # 如果文件大小符合预期，则输出找到并验证通过的消息
        print('Found and verified', filename)
    else:
        # 如果文件大小不符合预期，则输出错误消息并抛出异常
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + local_filename + '. Can you get to it with a browser?')
    # 返回下载后的文件路径
    return local_filename


# 调用maybe_download函数下载指定文件，并存储文件名
filename = maybe_download('text8.zip', 31344016)

# 输出下载后的文件名
print(filename)


# Read the data into a list of strings.
def read_data(filename):
    """
    从zip文件中的第一个文件读取文本数据，并将其作为单词列表返回。

    参数:
    filename: str, 指向zip文件的路径。

    返回:
    list, 从zip文件中提取的第一个文件的单词列表。
    """
    """将zip文件中的第一个文件作为单词列表提取出来"""
    with zipfile.ZipFile(filename) as f:
        # 读取zip文件中的第一个文件，将其转换为字符串，然后分割成单词列表
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


vocabulary = read_data(filename)
# 打印数据大小（即单词数量）
print('数据大小', len(vocabulary))

# 定义词汇表的大小，即保留最常用单词的数量
# 第二步：构建字典并将罕见词替换为UNK标记。
vocabulary_size = 50000


def build_dataset(words, n_words):
    """
    构建词汇表并转换单词到整数的索引。

    参数:
    words -- 单词列表
    n_words -- 构建的词汇表大小

    返回:
    data -- 单词索引列表
    count -- 单词出现次数的列表
    dictionary -- 单词到索引的字典
    reverse_dictionary -- 索引到单词的字典
    """
    """Process raw inputs into a dataset."""
    # 初始化词汇表，将未知单词（UNK）作为第一个元素
    count = [['UNK', -1]]
    # 根据出现频率添加除了最频繁的n_words - 1个单词之外的所有单词
    count.extend(collections.Counter(words).most_common(n_words - 1))
    # 创建字典，将单词映射到它们的索引
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    # 初始化数据列表，将每个单词替换为其索引
    data = list()
    # 初始化未知单词计数器
    unk_count = 0
    for word in words:
        # 查找单词的索引，如果不存在则使用UNK的索引
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    # 更新UNK单词出现次数
    count[0][1] = unk_count
    # 创建反向字典，将索引映射到单词
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


# 初始化词汇表和相关变量
# Filling 4 global variables:
# data - list of codes (integers from 0 to vocabulary_size-1).
#   This is the original text but words are replaced by their codes
# count - map of words(strings) to count of occurrences
# dictionary - map of words(strings) to their codes(integers)
# reverse_dictionary - maps codes(integers) to words(strings)
data, count, dictionary, reverse_dictionary = build_dataset(
    vocabulary, vocabulary_size)
del vocabulary  # 释放内存
# 打印最常见的单词和示例数据
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

# 初始化数据索引
data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    """
    生成训练批次的数据。

    这个函数从全局数据索引处开始，收集指定大小的批次数据及其对应的标签。
    它支持跳窗预测的训练方法，即在给定上下文单词的情况下，预测目标单词。

    参数:
    - batch_size: 每个批次生成的样本数量。
    - num_skips: 每个上下文窗口中采样的目标单词数量。
    - skip_window: 上下文窗口的大小。

    返回:
    - batch: 包含批次中所有样本的单词索引的数组。
    - labels: 包含批次中所有样本的目标单词索引的二维数组。
    """
    global data_index
    # 确保批次大小是跳过的倍数，且跳过次数不超过窗口大小的两倍
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    # 初始化批次数据和标签数组
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # 定义上下文窗口的大小
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    # 使用deque创建一个固定大小的缓冲区，用于存储窗口内的单词
    buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
    # 如果全局数据索引超出数据范围，则重置索引
    if data_index + span > len(data):
        data_index = 0
    # 将数据填充到缓冲区
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    # 为每个skip生成batch
    for i in range(batch_size // num_skips):
        # 生成除目标单词外的所有上下文单词的索引
        context_words = [w for w in range(span) if w != skip_window]
        # 从上下文单词中随机选择num_skips个作为目标单词
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            # 将目标单词填入批次数组
            batch[i * num_skips + j] = buffer[skip_window]
            # 将上下文单词填入标签数组
            labels[i * num_skips + j, 0] = buffer[context_word]
        # 如果全局数据索引超出数据范围，则重新填充缓冲区
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            # 否则，将下一个单词添加到缓冲区
            buffer.append(data[data_index])
            data_index += 1
    # 调整数据索引，以避免在批次末尾跳过单词
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


# 以下代码演示了如何使用generate_batch函数生成批次数据，并将其打印出来
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0],
          reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.
# 定义模型常量和参数
batch_size = 128
embedding_size = 128  # 嵌入向量的维度.
skip_window = 1  # 考虑左右各多少个词.
num_skips = 2  # 一个输入重复生成标签的次数.
num_sampled = 64  # 抽样负例的数量.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
# 定义验证集大小
valid_size = 16
# 定义验证集窗口大小
valid_window = 100
# 从验证窗口中随机选择16个样本，不重复
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
# 初始化TensorFlow图
# 第一步：构建图
graph = tf.Graph()

# 在默认的图中定义操作
with graph.as_default():
    # 定义输入数据的操作范围
    # Input data.
    with tf.name_scope('inputs'):
        # 定义训练输入的占位符，形状为[batch_size]
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        # 定义训练标签的占位符，形状为[batch_size, 1]
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        # 定义验证集数据集，作为常量
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # 指定操作在CPU上执行
    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # 定义嵌入层的操作范围
        # Look up embeddings for inputs.
        with tf.name_scope('embeddings'):
            # 初始化嵌入层权重，大小为[vocabulary_size, embedding_size]
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            # 通过输入查找嵌入向量
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # 定义负采样损失的权重和偏置
        # Construct the variables for the NCE loss
        with tf.name_scope('weights'):
            nce_weights = tf.Variable(
                tf.truncated_normal(
                    [vocabulary_size, embedding_size],
                    stddev=1.0 / math.sqrt(embedding_size)))
        with tf.name_scope('biases'):
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # 计算平均负采样损失
    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    # Explanation of the meaning of NCE loss:
    #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=embed,
                num_sampled=num_sampled,
                num_classes=vocabulary_size))
        # 将损失添加到摘要中
        tf.summary.scalar('loss', loss)

    # 构建使用梯度下降优化器，学习率为1.0
    # Construct the SGD optimizer using a learning rate of 1.0.
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # 计算嵌入向量的相似度
    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                              valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)

    # 合并所有摘要
    # Merge all summaries.
    merged = tf.summary.merge_all()

    # 初始化所有变量
    # Add variable initializer.
    init = tf.global_variables_initializer()

    # 创建一个保存器
    # Create a saver.
    saver = tf.train.Saver()

# 定义训练步数
# Step 5: Begin training.
num_steps = 100001

# 在给定的图中启动TensorFlow会话
with tf.Session(graph=graph) as session:
    # 创建一个写入器以将摘要写入日志目录，用于在TensorBoard中显示
    # 打开一个写入器来写入摘要。
    writer = tf.summary.FileWriter(FLAGS.log_dir, session.graph)

    # 初始化所有变量
    # 我们必须在使用变量之前初始化它们。
    init.run()
    print('已初始化')

    # 初始化平均损失
    average_loss = 0

    # 执行多个训练步骤
    for step in xrange(num_steps):
        # 生成一批训练数据
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        # 构建该批处理的feed字典
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # 定义运行元数据，用于可视化计算图
        # 定义元数据变量。
        run_metadata = tf.RunMetadata()

        # 运行优化操作并获取损失值和摘要
        # 我们通过评估优化器op执行一次更新步骤（将其包含在session.run()的返回值列表中）
        # 同时，评估merged op以从返回的“summary”变量中获取所有摘要。
        # 将元数据变量传递给会话，以便在TensorBoard中可视化计算图。
        _, summary, loss_val = session.run(
            [optimizer, merged, loss],
            feed_dict=feed_dict,
            run_metadata=run_metadata)
        # 更新平均损失
        average_loss += loss_val

        # 将返回的摘要添加到写入器中
        # 每个步骤都将返回的摘要添加到写入器中。
        writer.add_summary(summary, step)
        # 如果是最后一个步骤，将元数据添加到TensorBoard以可视化计算图
        # 为最后一次运行添加元数据以可视化图形。
        if step == (num_steps - 1):
            writer.add_run_metadata(run_metadata, 'step%d' % step)

        # 每2000步打印平均损失
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # 平均损失是对过去2000批处理损失的估计。
            print('第', step, '步的平均损失：', average_loss)
            average_loss = 0

        # 每10000步计算并打印相似性
        # 注意这很耗时（如果每500步计算，大约慢20％）
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # 最近邻的数量
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = '最接近%s的词：' % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)

    # 评估并保存最终嵌入
    final_embeddings = normalized_embeddings.eval()

    # 写入嵌入对应的词汇标签
    # 为嵌入写入相应的标签。
    with open(FLAGS.log_dir + '/metadata.tsv', 'w') as f:
        for i in xrange(vocabulary_size):
            f.write(reverse_dictionary[i] + '\n')

    # 保存模型
    # 保存模型以进行检查点。
    saver.save(session, os.path.join(FLAGS.log_dir, 'model.ckpt'))

    # 配置TensorBoard以可视化嵌入
    # 创建配置以在TensorBoard中可视化带有标签的嵌入。
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = embeddings.name
    embedding_conf.metadata_path = os.path.join(FLAGS.log_dir, 'metadata.tsv')
    projector.visualize_embeddings(writer, config)

    # 关闭写入器
    writer.close()


# Step 6: Visualize the embeddings.


# pylint: disable=missing-docstring
# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
    """
    在二维平面上绘制低维嵌入点并标注对应标签。

    参数:
    low_dim_embs: 二维数组，其中每一行代表一个高维对象的低维表示。
    labels: 列表，包含与低维表示对应的标签。
    filename: 字符串，指定保存图像的文件名。

    返回值:
    无
    """
    # 确保低维嵌入点的数量大于等于标签的数量
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'

    # 设置画布大小
    plt.figure(figsize=(18, 18))  # in inches

    # 遍历每个低维表示及其对应的标签
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom')

    # 保存图像
    plt.savefig(filename)


# 尝试导入机器学习和可视化库，用于二维降维和绘图
try:
    # pylint: disable=g-import-not-at-top
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # 初始化TSNE模型，用于降维
    # 参数perplexity定义了模型的不确定性，n_components指定降维后的维度
    # init参数指定初始化方法，n_iter指定迭代次数，method指定使用的算法
    tsne = TSNE(
        perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')

    # 定义只绘制前500个样本
    plot_only = 500

    # 对预训练的嵌入进行降维处理
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])

    # 从索引获取对应的标签
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]

    # 绘制降维后的数据及其标签
    plot_with_labels(low_dim_embs, labels, 'tsne.png')

# 如果导入失败，则提示用户安装相关库
except ImportError as ex:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')
    print(ex)
