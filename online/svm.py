# coding=utf-8

# 输入一个string 首先分词，然后读取本地word2vec模型，计算向量,
# 读取本地的svm模型，做预测
# 返回预测结果
import warnings

warnings.filterwarnings("ignore")
import jieba
import numpy as np
from numpy import *

try:
    import cPickle as pickle
except:
    import pickle


def kernel(x, y, sigma):
    x = mat(x)
    y = mat(y)
    temp = x - y
    return math.exp(temp * temp.T / (-2) * sigma * sigma)


def label(x):
    alphs_result = pickle.loads(open("data/svm_model_1", 'rb').read(), encoding='iso-8859-1')
    x_result = pickle.loads(open("data/svm_model_2", 'rb').read(), encoding='iso-8859-1')
    y_result = pickle.loads(open("data/svm_model_3", 'rb').read(), encoding='iso-8859-1')
    b = pickle.loads(open("data/svm_model_4", 'rb').read(), encoding='iso-8859-1')

    num = len(alphs_result)
    re = 0.0
    for i in range(num):
        re += alphs_result[i] * y_result[i] * kernel(x_result[i], x, 1)
    re += b
    if (re < 0):
        return 0
    else:
        return 1


def feature(word_model, sentence):
    seg_list = jieba.cut(sentence, cut_all=False)
    count = 0
    for j in seg_list:
        if j not in word_model:
            continue
        if count == 0:
            old = word_model[j]
            new = np.zeros(shape=old.shape)
        else:
            new = word_model[j]
        old = old + new
        count += 1
    if (count != 0):
        old = old * (1.0 / count)
        return list(old)


def svm_predict(word_model, sentence):
    x = feature(word_model, sentence)
    return label(x)
