from bayes import *
import math


def runBayes(trainSet):
    list = {}
    N = [0, 0]  # 样本数量
    T = [0, 0]  # 词条数量
    for line in trainSet:
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


# 在测试数据上测试效果，返回P,R,F
def runTest(testSet, condprob, prior):
    posRight = 0  # 正常短信判断正确
    negRight = 0  # 垃圾短信判断正确
    posWrong = 0  # 正常短信判断错误
    negWrong = 0  # 垃圾短信判断错误

    for line in testSet:
        real = line[0]
        score = [i for i in prior]
        for word in line.split()[1:]:
            if word in condprob.keys():
                score[0] += condprob[word][0]
                score[1] += condprob[word][1]

        if score[0] < score[1]:
            result = "1"
        else:
            result = "0"
        if real == "0" and result == "0": negRight += 1
        if real == "0" and result == "1": negWrong += 1
        if real == "1" and result == "0": posWrong += 1
        if real == "1" and result == "1": posRight += 1

    P = posRight / (posRight + negWrong)
    R = posRight / (posRight + posWrong)
    F = 2 * P * R / (P + R)

    return P, R, F


def crossValidation(k):
    lines = []
    with open("words", encoding='utf-8') as file:
        for line in file:
            lines.append(line)
    testLength = int(len(lines) / k)

    Fs = 0.0
    Ps = 0.0
    Rs = 0.0

    for i in range(k):
        start = i * testLength
        end = start + testLength

        testSet = lines[start:end]
        trainSet = lines[0:start]
        trainSet.extend(lines[end:len(lines)])

        prior, condprob = runBayes(trainSet)
        P, R, F = runTest(testSet, condprob, prior)

        print("第%d次验证，测试样本为[%d:%d]，本次结果:" % (i, start, end))
        print("P: %f" % P)
        print("R: %f" % R)
        print("F: %f" % F)

        Ps += P
        Rs += R
        Fs += F

    print("完成%d折交叉验证，结果如下: " % k)
    print("平均P: %f" % (Ps / k))
    print("平均R: %f" % (Rs / k))
    print("平均F: %f" % (Fs / k))


if __name__ == '__main__':
    crossValidation(10)
