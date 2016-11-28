# coding=utf-8

import jieba
import math
import time

# 分词文件路径
words_data_path = "data/words.txt"


# 朴素贝叶斯（多项式模型）
def runBayes():
    list = {}
    N = [0, 0]  # 样本数量
    T = [0, 0]  # 词条数量
    with open(words_data_path, 'r', encoding='utf-8') as file:
        for line in file:
            words = line.split()
            c = int(words[0])  # 第一个字符为类别
            N[c] += 1
            for word in words[1:]:
                T[c] += 1
                if word not in list.keys(): list[word] = [0, 0]
                list[word][c] += 1

    prior = [math.log(x / sum(N)) for x in N]
    condprob = {}

    for word in list.keys():
        condprob[word] = [math.log((list[word][0] + 1) / (T[0] + len(list))), \
                          math.log((list[word][1] + 1) / (T[1] + len(list)))]
    return prior, condprob


# 预测
def predict(line, prior, condprob):
    score = [i for i in prior]
    for word in jieba.cut(line):
        if word in condprob.keys():
            score[0] += condprob[word][0]
            score[1] += condprob[word][1]
    if score[0] > score[1]:
        return 0
    else:
        return 1
