import numpy as np
import re
import random


# 创建词汇表
def create_word_list(data_set):
    # data_set:包含多个文档的数据集
    vocab_set = set([])  # 去重词汇表
    for document in data_set:
        vocab_set = vocab_set | set(document)  # 取并集
    return list(vocab_set)


# 词集模型
def set_of_words(vocab_list, input_set):
    # vocab_list:去重词汇表
    # input_set：输入文档
    return_vec = [0] * len(vocab_list)  # 建立和去重词汇表相同长度的全0向量
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1  # 遍历所有单词，如果存在使相应向量=1
        else:
            print("妹这个单词哦~")
    return return_vec


# 词袋模型
def bag_of_words(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
    return return_vec


# 训练朴素贝叶斯
def train_BYS(train_matrix, train_classes):
    # train_matrix:每个returnVec组成的矩阵
    # train_classes：每个reVec对应的类别，1侮辱类 0正常类

    num_train_docs = len(train_matrix)  # 总文档数
    num_words = len(train_matrix[0])  # 每个文档总字数
    p_1 = sum(train_classes) / float(num_train_docs)  # 文档属于侮辱类
    # 拉普拉斯修正_防止下溢出
    p0_num = np.ones(num_words)  # 分子各单词出现数初始化为1
    p1_num = np.ones(num_words)
    p0_denom = 2.0  # 分母总单词数初始化为类别数 2
    p1_denom = 2.0

    for i in range(num_train_docs):  # 遍历训练样本
        if train_classes[i] == 1:
            p1_num += train_matrix[i]  # 侮辱类各个单词数量
            p1_denom += sum(train_matrix[i])  # 侮辱类总单词数量
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])

    p1_vect = np.log(p1_num / p1_denom)  # 取对数
    p0_vect = np.log(p0_num / p0_denom)
    # print(p0Vect)
    return p0_vect, p1_vect, p_1


def classify_NB(vec2_classify, p0_vec, p1_vec, p_class1):
    """
    对测试文档进行分类

    Parameter:
    vec2Classify:测试文档向量
    p0Vec:正常类中每一个词的条件概率
    p1Vec:侮辱类中每一个词的条件概率
    pClass1:侮辱类占总样本概率
    """
    p1 = sum(vec2_classify * p1_vec) + np.log(p_class1)
    p0 = sum(vec2_classify * p0_vec) + np.log(1.0 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


# 将字符串转换为小写字符列表
def text_parse(bigString):
    """
    Parameter:
    bigString:输入字符串

    Return：
    tok.lower()：小写字符列表
    """

    # 将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    list_of_tokens = re.split(r'\W+', bigString)
    # 除了单个字母，其它单词变成小写
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


# 测试朴素bys分类器
def spam_test(method='bag'):
    # method有bag词袋和set词集两中
    if method == 'bag':  # 判断使用词袋模型还是词集模型
        words2Vec = bag_of_words
    elif method == 'set':
        words2Vec = set_of_words

    doc_list = []
    class_list = []

    # 遍历文件夹
    for i in range(1, 26):
        # 读取垃圾邮件 转化成字符串列表
        word_list = text_parse(
            open('D:/python/project/study/study/贝叶斯/data/spam/%d.txt' % i, 'r').read())
        # 将列表记录加入文档列表并分类为1 侮辱类
        doc_list.append(word_list)
        class_list.append(1)
        # 读取正常文件
        word_list = text_parse(
            open('D:/python/project/study/study/贝叶斯/data/ham/%d.txt' % i, 'r').read())
        doc_list.append(word_list)
        class_list.append(0)

    # 创建去重词汇表
    vocab_list = create_word_list(doc_list)
    # print(vocabList)

    # 创建索引
    trainSet = list(range(50))
    testSet = []

    # 分割测试急
    for i in range(10):
        # 从索引中随机抽取十个并从索引中删除十个
        randIndex = int(random.uniform(0, len(trainSet)))
        testSet.append(trainSet[randIndex])
        del (trainSet[randIndex])

    # 创建训练集矩阵和类别向量
    trainMat = []
    trainClasses = []

    for docIndex in trainSet:
        # 将生成模型添加到训练矩阵并记录类别
        trainMat.append(words2Vec(vocab_list, doc_list[docIndex]))
        trainClasses.append(class_list[docIndex])

    # 训练bys
    p0V, p1V, pSpam = train_BYS(np.array(trainMat), np.array(trainClasses))

    # 错误分类器
    error = 0

    for docIndex in testSet:
        word_vec = words2Vec(vocab_list, doc_list[docIndex])
        # 测试
        if classify_NB(np.array(word_vec), p0V, p1V, pSpam) != class_list[docIndex]:
            error += 1
            # print("分类错误的：",docList[docIndex])
    # print("错误率：%.2f%%"%(float(error)/len(testSet)*100))
    error_rate = float(error) / len(testSet)  # 分类错误率
    return error_rate


if __name__ == "__main__":
    total = 100
    print('使用词袋模型训练：')
    sum_bag_error = 0
    for i in range(total):
        sum_bag_error += spam_test(method='bag')
    print('使用词袋模型训练' + str(total) + '次得到的平均错误率为： ' + str((sum_bag_error / total)))
