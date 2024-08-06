# coding=utf-8
# 打印文档字符串，提供程序的基本描述
print(__doc__)

# 导入numpy库，用于处理数值计算
import numpy as np
# 导入warnings模块，用于控制警告信息
import warnings

# 忽略所有警告信息，避免干扰
warnings.filterwarnings("ignore")
# 导入hmmlearn库中的hmm模块，用于实现隐马尔可夫模型(HMM)
from hmmlearn import hmm
# 导入自定义的模块，用于获取HMM模型的参数
import get_hmm_param as pa

# 从hmmlearn.hmm中导入MultinomialHMM类，用于创建多项分布的HMM模型
from hmmlearn.hmm import MultinomialHMM as mhmm

# 获取并打印HMM模型的初始状态概率分布
startprob = np.array(pa.get_startprob())
print("startprob is ", startprob)
# 获取并打印HMM模型的状态转移矩阵
transmat = np.array(pa.get_transmat())
print("transmat is ", transmat)
# 获取并打印HMM模型的发射概率矩阵
emissionprob = np.array(pa.get_emissionprob())
print("emmissionprob is ", emissionprob)
# 创建一个具有4个隐藏状态的多项分布HMM模型
mul_hmm = mhmm(n_components=4)

# 设置HMM模型的初始状态概率分布
mul_hmm.startprob_ = startprob

# 设置HMM模型的状态转移矩阵
mul_hmm.transmat_ = transmat

# 设置HMM模型的发射概率矩阵
mul_hmm.emissionprob_ = emissionprob

# 定义一个中文短语，用于HMM模型的预测
phase = u"我要吃饭谢天谢地"

# 根据定义的短语，获取对应的数值数组，并重塑为每个字符一个样例的形式
X = np.array(pa.get_array_from_phase(phase))
X = X.reshape(len(phase), 1)
print("X is ", X)

# 使用HMM模型对输入的短语进行预测，返回每个字符对应的隐藏状态
Y = mul_hmm.predict(X)
print("Y is ", Y)
# {B（词开头），M（词中），E（词尾），S（独字词）} {0,1,2,3}
