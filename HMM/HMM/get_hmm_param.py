# coding=utf8
from data import data
import json
import logging


def prints(s):
    # 打印函数，本示例中未实现具体功能
    pass
    print(s)


def get_startprob():
    """获取BMES矩阵"""
    c = 0
    c_map = {"B": 0, "M": 0, "E": 0, "S": 0}
    # 计算每个状态的出现次数
    for v in data:
        for key in v:
            value = v[key]
        c = c + 1
        prints("value[0] is " + value[0])
        c_map[value[0]] = c_map[value[0]] + 1
        prints("c_map[value[0]] is " + str(c_map[value[0]]))
    # 计算每个状态的概率
    res = []
    for i in "BMES":
        res.append(c_map[i] / float(c))
    return res


def get_transmat():
    """获取状态之间的转移矩阵"""
    c = 0
    c_map = {}
    for v in data:
        for key in v:
            value = v[key]

        prints("value[0] is " + value[0])
        # 记录每对状态的转移次数
        for v_i in range(len(value) - 1):
            couple = value[v_i:v_i + 2]
            c_couple_source = c_map.get(couple, 0)
            c_map[couple] = c_couple_source + 1
            c = c + 1
    # 计算转移概率
    res = []
    for i in "BMES":
        col = []
        col_count = 0
        for j in "BMES":
            col_count = c_map.get(i + j, 0) + col_count

        for j in "BMES":
            col.append(c_map.get(i + j, 0) / float(col_count))
        res.append(col)
    return res


def get_words():
    """获取示例词汇"""
    return u"我要吃饭天气不错谢天地"


def get_word_map():
    """建立词汇与索引的映射"""
    words = get_words()
    res = {}
    for i in range(len(words)):
        res[words[i]] = i
    return res


def get_array_from_phase(phase):
    """从阶段获取数组表示"""
    word_map = get_word_map()
    res = []
    for key in phase:
        res.append(word_map[key])
    return res


def get_emissionprob():
    """获取状态与观测值之间的发射概率矩阵"""
    c = 0
    c_map = {}
    for v in data:
        for key in v:
            k = key
            value = v[key]

        prints("value[0] is " + value[0])
        # 记录每个状态对应观测值的出现次数
        for v_i in range(len(value)):
            couple = value[v_i] + k[v_i]
            prints("emmition's couple is " + couple)
            c_couple_source = c_map.get(couple, 0)
            c_map[couple] = c_couple_source + 1
            c = c + 1
    # 计算发射概率
    res = []
    prints("emmition's c_map is " + str(c_map))
    words = get_words()
    for i in "BMES":
        col = []
        for j in words:
            col.append(c_map.get(i + j, 0) / float(c))
        res.append(col)
    return res


if (__name__ == "__main__"):
    # 执行主程序时，输出结果
    pass
    # print("startprob is ",get_startprob())
    print("transmat is ", get_transmat())
    # print("emissionprob is " , get_emissionprob())
    # print("word map is ",get_word_map())
