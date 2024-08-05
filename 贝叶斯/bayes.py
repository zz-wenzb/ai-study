import numpy as np
import re
import random


def textParse(input_string):
    """
    解析文本，将输入字符串分解成单词列表。

    参数:
    input_string: 一个字符串，包含需要解析的文本。

    返回:
    一个单词列表，其中每个单词都是input_string中的一个连续的字母序列。
    """
    listofTokens = re.split(r'\W+', input_string)
    return [tok.lower() for tok in listofTokens if len(tok) > 2]


def creatVocablist(doclist):
    """
    创建词汇列表，包含所有文档中的不重复单词。
    把所有文档的单词并集后去重

    参数:
    doclist: 一个文档列表，每个文档都是一个单词列表。

    返回:
    一个包含所有不重复单词的列表。
    """
    vocabSet = set([])
    for document in doclist:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWord2Vec(vocablist, inputSet):
    """
    将单词列表转换为词向量。

    参数:
    vocablist: 词汇列表 （所有文档单词并集去重）。
    inputSet: 输入的单词列表 （某个文档下的词列表）。

    返回:
    一个词向量，其中每个元素表示词汇列表中的一个单词是否出现在输入列表中。
    """
    returnVec = [0] * len(vocablist)
    for word in inputSet:
        if word in vocablist:
            returnVec[vocablist.index(word)] = 1
    return returnVec


def trainNB(trainMat, trainClass):
    """
    训练朴素贝叶斯分类器。

    参数:
    trainMat: 训练文档的词向量矩阵。
    trainClass: 训练文档的类别标签列表。

    返回:
    p0Vec: 类别0中各单词的条件概率向量。
    p1Vec: 类别1中各单词的条件概率向量。
    p1: 类别1的先验概率。
    """
    numTrainDocs = len(trainMat)
    numWords = len(trainMat[0])
    p1 = sum(trainClass) / float(numTrainDocs)
    p0Num = np.ones((numWords))
    p1Num = np.ones((numWords))
    p0Denom = 2
    p1Denom = 2

    for i in range(numTrainDocs):
        if trainClass[i] == 1:
            p1Num += trainMat[i]
            p1Denom += sum(trainMat[i])
        else:
            p0Num += trainMat[i]
            p0Denom += sum(trainMat[i])
    p1Vec = np.log(p1Num / p1Denom)
    p0Vec = np.log(p0Num / p0Denom)
    return p0Vec, p1Vec, p1


def classifyNB(wordVec, p0Vec, p1Vec, p1_class):
    """
    使用朴素贝叶斯分类器进行分类。

    参数:
    wordVec: 待分类的文档词向量。
    p0Vec: 类别0中各单词的条件概率向量。
    p1Vec: 类别1中各单词的条件概率向量。
    p1_class: 类别1的先验概率。

    返回:
    0或1，表示分类结果。
    """
    p1 = np.log(p1_class) + sum(wordVec * p1Vec)
    p0 = np.log(1.0 - p1_class) + sum(wordVec * p0Vec)
    if p0 > p1:
        return 0
    else:
        return 1


def spam():
    """
    垃圾邮件分类器的主函数。
    """
    doclist = []
    classlist = []
    for i in range(1, 26):
        wordlist = textParse(open('data/spam/%d.txt' % i, 'r').read())
        doclist.append(wordlist)
        classlist.append(1)

        wordlist = textParse(open('data/ham/%d.txt' % i, 'r').read())
        doclist.append(wordlist)
        classlist.append(0)
    # 所有文档单词并集去重后的列表
    vocablist = creatVocablist(doclist)
    trainSet = list(range(50))
    testSet = []

    # 随机选10个数
    # 区分训练集和测试集
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainSet)))
        testSet.append(trainSet[randIndex])
        del (trainSet[randIndex])

    trainMat = []
    trainClass = []
    for docIndex in trainSet:
        trainMat.append(setOfWord2Vec(vocablist, doclist[docIndex]))
        trainClass.append(classlist[docIndex])

    p0Vec, p1Vec, p1 = trainNB(np.array(trainMat), np.array(trainClass))
    errorCount = 0
    for docIndex in testSet:
        wordVec = setOfWord2Vec(vocablist, doclist[docIndex])
        if classifyNB(np.array(wordVec), p0Vec, p1Vec, p1) != classlist[docIndex]:
            errorCount += 1
    print('当前10个测试样本，错了：', errorCount)


if __name__ == '__main__':
    spam()
