# coding=utf-8

import jieba
import re
import math
import time

# 停用词路径
stop_word_path = "stopwords.txt"
# 原始数据文件
source_data_path = "带标签短信.txt"
# 分词文件路径
words_data_path = "words.txt"
# 测试数据文件
test_data_path = "测试短信.txt"


# 读取停用词
def getStopWords():
    words = []
    with open(stop_word_path, encoding='utf-8') as file:
        for line in file.readlines():
            words.append(line.strip('\n'))
    return set(words)


# 将短信数据分词
def segmentWords():
    stopwords = getStopWords()

    startTime = time.time()
    file = open(source_data_path, 'r', encoding='utf-8')
    words = []
    for line in file:
        fenci = filter(lambda x: x not in stopwords, jieba.cut(line[1:].strip()))
        word = line[0] + ' ' + ' '.join(fenci) + '\n'
        words.append(word)
        print(word)

    save = open(words_data_path, 'w', encoding='utf-8')
    save.writelines(words)
    endTime = time.time()
    print("分词时间: %f" % (endTime - startTime))


# 在测试集上测试
def runTest(prior, condprob):
    startTime = time.time()

    posRight = 0
    negRight = 0
    posWrong = 0
    negWrong = 0

    file = open(test_data_path, 'r', encoding='utf-8')
    for line in file:
        real = int(line[0])
        result = predict(jieba.cut(line[1:]), prior, condprob)
        if real == 0 and result == 0: negRight += 1
        if real == 0 and result == 1: negWrong += 1
        if real == 1 and result == 0: posWrong += 1
        if real == 1 and result == 1: posRight += 1
    file.close()

    P = posRight / (posRight + negWrong)
    R = posRight / (posRight + posWrong)

    F = 2 * P * R / (P + R)

    print("P：%f" % P)
    print("R: %f" % R)
    print("F: %f" % F)

    print("垃圾短信总数: %d" % (posWrong + posRight))
    print("正常短信总数: %d" % (negRight + negWrong))
    print("正确判断垃圾短信数: %d" % posRight)
    print("正确判断正常短信数: %d" % negRight)

    endTime = time.time()
    print("分词时间: %f" % (endTime - startTime))


# 朴素贝叶斯（多项式模型）
def runBayes():
    startTime = time.time()
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
    endTime = time.time()
    print("读词时间: %f" % (endTime - startTime))

    for word in list.keys():
        condprob[word] = [math.log((list[word][0] + 1) / (T[0] + len(list))), \
                          math.log((list[word][1] + 1) / (T[1] + len(list)))]
    print("计算时间: %f" % (time.time() - endTime))
    return prior, condprob


# 预测，words为词列表
def predict(words, prior, condprob):
    score = [i for i in prior]
    for word in words:
        if word in condprob.keys():
            score[0] += condprob[word][0]
            score[1] += condprob[word][1]
    if score[0] > score[1]:
        return 0
    else:
        return 1


if __name__ == '__main__':
    # 分词
    # segmentWords()

    # 训练模型
    prior, condprob = runBayes()

    # 测试
    runTest(prior, condprob)

    # 检测一条短信
    result = predict("好消息，店庆大酬宾，一律五折！", prior, condprob)
    print(result)
