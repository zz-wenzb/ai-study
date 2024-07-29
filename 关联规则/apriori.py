def loadDataSet():
    """
    加载数据集。

    返回:
    - 一个列表的列表，每个子列表代表一个事务。
    """
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createC1(dataSet):
    """
    从数据集中创建候选项集C1。

    参数:
    - dataSet: 数据集，一个列表的列表。

    返回:
    - C1: 候选项集，一个列表的列表，每个元素是数据集中出现的单独项。
    """
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))


def scanD(D, CK, minSupport):
    """
    在数据集中扫描候选项集，以找到支持度大于等于minSupport的项集。

    参数:
    - D: 数据集，一个列表的列表。
    - CK: 候选项集，一个列表的frozenset。
    - minSupport: 最小支持度阈值。

    返回:
    - 两个值的元组：一个列表，包含支持度大于等于minSupport的项集；一个字典，包含每个项集的支持度。
    """
    ssCnt = {}
    for tid in D:
        for can in CK:
            if can.issubset(tid):
                if not can in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(list(D)))
    retlist = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retlist.insert(0, key)
        supportData[key] = support
    return retlist, supportData


def aprioriGen(LK, k):
    """
    根据Lk生成下一个候选项集Ck。

    参数:
    - LK: 长度为k-1的频繁项集列表。
    - k: 生成的候选项集的长度。

    返回:
    - Ck: 长度为k的候选项集列表。
    """
    retlist = []
    lenLK = len(LK)
    for i in range(lenLK):
        for j in range(i + 1, lenLK):
            L1 = list(LK[i])[:k - 2]
            L2 = list(LK[j])[:k - 2]
            if L1 == L2:
                retlist.append(LK[i] | LK[j])
    return retlist


def apriori(dataSet, minSupport=0.5):
    """
    执行Apriori算法，找出数据集中频繁项集和支持度。

    参数:
    - dataSet: 数据集，一个列表的列表。
    - minSupport: 最小支持度阈值。

    返回:
    - 两个值的元组：一个列表，包含不同长度的频繁项集；一个字典，包含每个频繁项集的支持度。
    """
    C1 = createC1(dataSet)
    L1, supportData = scanD(dataSet, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k - 2]) > 0):
        CK = aprioriGen(L[k - 2], k)
        LK, supk = scanD(dataSet, CK, minSupport)
        supportData.update(supk)
        L.append(LK)
        k += 1
    return L, supportData


def generateRules(L, supportData, minConf=0.6):
    """
    根据频繁项集L和支持度数据生成关联规则。

    参数:
    - L: 频繁项集列表。
    - supportData: 支持度数据字典。
    - minConf: 最小置信度阈值。

    返回:
    - 一个列表，包含满足最小置信度的关联规则。
    """
    rulelist = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            rulessFromConseq(freqSet, H1, supportData, rulelist, minConf)


def rulessFromConseq(freqSet, H, supportData, rulelist, minConf=0.6):
    """
    从频繁项集的序列中生成关联规则。

    参数:
    - freqSet: 频繁项集。
    - H: 一个候选项集列表，用于生成规则。
    - supportData: 支持度数据字典。
    - rulelist: 关联规则列表。
    - minConf: 最小置信度阈值。
    """
    m = len(H[0])
    while (len(freqSet) > m):
        H = calConf(freqSet, H, supportData, rulelist, minConf)
        if (len(H) > 1):
            aprioriGen(H, m + 1)
            m += 1
        else:
            break


def calConf(freqSet, H, supportData, rulelist, minConf=0.6):
    """
    计算候选项集的置信度，并根据最小置信度阈值进行剪枝。

    参数:
    - freqSet: 频繁项集。
    - H: 一个候选项集列表。
    - supportData: 支持度数据字典。
    - rulelist: 关联规则列表。
    - minConf: 最小置信度阈值。

    返回:
    - 一个列表，包含满足最小置信度的候选项集。
    """
    prunedh = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            rulelist.append((freqSet - conseq, conseq, conf))
            prunedh.append(conseq)
    return prunedh


if __name__ == '__main__':
    dataSet = loadDataSet()
    L, support = apriori(dataSet)
    i = 0
    for freq in L:
        print('项数', i + 1, ':', freq)
        i += 1
    rules = generateRules(L, support, minConf=0.5)
