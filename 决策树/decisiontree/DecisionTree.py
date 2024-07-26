# -*- coding: UTF-8 -*-
# 导入matplotlib字体管理器，用于在图表中设置中文字体
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
# 导入math库的log函数，用于计算对数
from math import log
# 导入operator库的itemgetter函数，用于获取字典项排序
import operator


def createDataSet():
    """
        创建一个简单数据集，包含特征和对应的标签。

        返回:
        - dataSet: 二维列表，每一行代表一个样本，最后一列是标签。
        - labels: 列表，包含特征列的标签。
        """
    dataSet = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['F1-AGE', 'F2-WORK', 'F3-HOME', 'F4-LOAN']
    return dataSet, labels


def createTree(dataset, labels, featLabels):
    """
       递归构建决策树。

       参数:
       - dataset: 用于训练的样本数据集。
       - labels: 特征列的标签。
       - featLabels: 已经使用的特征标签。

       返回:
       - myTree: 构建好的决策树。
       """
    classList = [example[-1] for example in dataset]
    # 如果所有样本属于同一类，则返回该类
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果数据集中只有一个特征，则返回该特征值出现最多的类别
    if len(dataset[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataset)
    bestFeatLabel = labels[bestFeat]
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel: {}}
    del labels[bestFeat]
    featValue = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValue)
    for value in uniqueVals:
        sublabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataset, bestFeat, value), sublabels, featLabels)
    return myTree


def majorityCnt(classList):
    """
       计算列表中出现次数最多的元素。

       参数:
       - classList: 包含元素的列表。

       返回:
       - 出现次数最多的元素。
       """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedclassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedclassCount[0][0]


def chooseBestFeatureToSplit(dataset):
    """
        根据信息增益选择最佳分割特征。

        参数:
        - dataset: 用于训练的样本数据集。

        返回:
        - 最佳特征的索引。
        """
    numFeatures = len(dataset[0]) - 1
    baseEntropy = calcShannonEnt(dataset)
    bestInfoGain = 0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList)
        newEntropy = 0
        for val in uniqueVals:
            subDataSet = splitDataSet(dataset, i, val)
            prob = len(subDataSet) / float(len(dataset))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def splitDataSet(dataset, axis, val):
    """
       根据指定特征值分割数据集。

       参数:
       - dataset: 待分割的数据集。
       - axis: 特征索引。
       - val: 特征值。

       返回:
       - 分割后的数据集。
       """
    retDataSet = []
    for featVec in dataset:
        if featVec[axis] == val:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def calcShannonEnt(dataset):
    """
        计算数据集的香农熵。

        参数:
        - dataset: 待计算香农熵的数据集。

        返回:
        - 香农熵的值。
        """
    numexamples = len(dataset)
    labelCounts = {}
    for featVec in dataset:
        currentlabel = featVec[-1]
        if currentlabel not in labelCounts.keys():
            labelCounts[currentlabel] = 0
        labelCounts[currentlabel] += 1

    shannonEnt = 0
    for key in labelCounts:
        prop = float(labelCounts[key]) / numexamples
        shannonEnt -= prop * log(prop, 2)
    return shannonEnt


def getNumLeafs(myTree):
    """
       计算决策树的叶子节点数量。

       参数:
       - myTree: 待计算的决策树。

       返回:
       - 叶子节点的数量。
       """
    numLeafs = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    """
       计算决策树的深度。

       参数:
       - myTree: 待计算的决策树。

       返回:
       - 决策树的深度。
       """
    maxDepth = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """
       在图表上绘制决策树的节点。

       参数:
       - nodeTxt: 节点文本。
       - centerPt: 节点中心位置。
       - parentPt: 父节点位置。
       - nodeType: 节点类型（决策节点或叶子节点）。
       """
    arrow_args = dict(arrowstyle="<-")
    font = FontProperties(fname=r"c:\windows\fonts\simsunb.ttf", size=14)
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def plotMidText(cntrPt, parentPt, txtString):
    """
       在父节点和子节点之间绘制文本。

       参数:
       - cntrPt: 子节点中心位置。
       - parentPt: 父节点位置。
       - txtString: 要绘制的文本。
       """
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
    """
        在图表上绘制整个决策树。

        参数:
        - myTree: 待绘制的决策树。
        - parentPt: 父节点位置。
        - nodeTxt: 父节点文本。
        """
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")
    leafNode = dict(boxstyle="round4", fc="0.8")
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = next(iter(myTree))
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    """
    创建并显示一个决策树的图形表示。

    该函数利用matplotlib库绘制给定决策树的结构和相关信息。

    参数:
    inTree -- 决策树的数据结构，用于绘图。
    """
    # 初始化绘图窗口，设置背景色为白色
    fig = plt.figure(1, facecolor='white')  # 创建fig
    # 清空当前图形，为绘制新图做准备
    fig.clf()  # 清空fig
    # 定义子图属性，隐藏刻度标记
    axprops = dict(xticks=[], yticks=[])
    # 创建一个无边框的子图，并存储为全局变量ax1，用于绘制决策树
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # 去掉x、y轴
    # 计算并存储决策树的叶子节点数量，用于后续的图形布局
    plotTree.totalW = float(getNumLeafs(inTree))  # 获取决策树叶结点数目
    # 计算并存储决策树的深度，用于后续的图形布局
    plotTree.totalD = float(getTreeDepth(inTree))  # 获取决策树层数
    # 初始化x轴偏移量，用于调整节点在x轴上的位置
    plotTree.xOff = -0.5 / plotTree.totalW;
    # 初始化y轴偏移量，用于调整节点在y轴上的位置
    plotTree.yOff = 1.0;  # x偏移
    # 开始绘制决策树，此处调用递归函数plotTree进行绘制
    plotTree(inTree, (0.5, 1.0), '')  # 绘制决策树
    # 显示绘制的图形
    plt.show()


if __name__ == '__main__':
    dataset, labels = createDataSet()
    featLabels = []
    myTree = createTree(dataset, labels, featLabels)
    createPlot(myTree)
