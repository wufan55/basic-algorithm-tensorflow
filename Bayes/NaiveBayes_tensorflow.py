import tensorflow as tf
import numpy as np


def loadDataSet():
    """
    加载数据集

    :return trainMat: type=list, shape=(40, 692), 训练集
    :return trainClass: type=list, shape=(40,), 训练集标签
    :return testMat: type=list, shape=(10, 692), 测试集
    :return testClass: type=list, shape=(10,), 测试集标签
    """

    def createVocabList(dataSet):
        vocabSet = set([])
        for docment in dataSet:
            # 两个集合的并集
            vocabSet = vocabSet | set(docment)
            # 转换成列表
        return list(vocabSet)

    def textParse(bigString):
        import re
        listOfTokens = re.split(r'\W+', bigString)
        return [tok.lower() for tok in listOfTokens if len(tok) > 2]

    def bagOfWords2Vec(vocabList, inputSet):
        # vocablist为词汇表，inputSet为输入的邮件
        returnVec = [0] * len(vocabList)
        for word in inputSet:
            if word in vocabList:
                # 查找单词的索引
                returnVec[vocabList.index(word)] = 1
        return returnVec

    docList = []
    classList = []
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i,encoding="ISO-8859-1").read())
        docList.append(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i,encoding="ISO-8859-1").read())
        docList.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)

    trainSet = np.random.choice(50, 40, replace=False)
    testSet = np.setdiff1d(np.array(range(50)), trainSet)

    trainMat = []
    trainClass = []
    testMat = []
    testClass = []
    for docIndex in trainSet:
        trainMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))
        trainClass.append(classList[docIndex])
    for docIndex in testSet:
        testMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))
        testClass.append(classList[docIndex])

    return np.array(trainMat), np.array(trainClass, dtype=float), np.array(testMat), np.array(testClass, dtype=float)


def bayes(trainMat, trainClass):
    """
    计算贝叶斯概率

    :param trainMat: shape=(40, 692), 训练数据集
    :param trainClass: shape=(40,), 训练数据集标签

    :return pOneWords: type=tensor, shape=(1, 692), 类别为1中各单词的出现概率
    :return pZeroWords: type=tensor, shape=(1, 692), 类别为0中各单词的出现概率
    :return pClassOne: type=tensor, shape=(1,), 类别为1的概率
    :return pClassZero: type=tensor, shape=(1,), 类别为0的概率
    """

    # 获取训练数据个数, numTrain=40
    numTrain = trainMat.get_shape().as_list()[0]
    # 获取训练数据的值个数, numWords=692
    numWords = trainMat.get_shape().as_list()[1]

    # 计算类别为1的概率
    pClassOne = tf.divide(tf.reduce_sum(trainClass, keepdims=True), tf.cast(numTrain, dtype=tf.float32))
    # 计算类别为0的概率
    pClassZero = 1 - pClassOne

    # 统计各类中各单词的数量, shape=(1, numWords)
    pOneWordsNum = tf.reduce_sum(tf.multiply(trainMat, tf.tile(tf.expand_dims(trainClass, 1), [1, numWords])), axis=0, keepdims=True)
    pZeroWordsNum = tf.reduce_sum(tf.multiply(trainMat, tf.tile(tf.expand_dims(1-trainClass, 1), [1, numWords])), axis=0, keepdims=True)

    # 统计各类中单词的总数, shape=(1, 1)
    pOneWordsTotal = tf.reduce_sum(pOneWordsNum, axis=1, keepdims=True)
    pZeroWordsTotal = tf.reduce_sum(pZeroWordsNum, axis=1, keepdims=True)

    # 计算各类中各单词出现的概率, shape = (1, numWords)
    pOneWords = tf.divide(pOneWordsNum, tf.tile(pOneWordsTotal, [1, numWords]))
    pZeroWords = tf.divide(pZeroWordsNum, tf.tile(pZeroWordsTotal, [1, numWords]))

    return pOneWords, pZeroWords, pClassOne, pClassZero


def classify(testMat, p1Vec, p0Vec, pClass1, pClass0):
    """
    对测试集进行朴素贝叶斯分类

    :param testMat: shape=(10, 692), 测试数据集
    :param p1Vec: type=tensor, shape=(1, 692), 类别为1中各单词的出现概率
    :param p0Vec: type=tensor, shape=(1, 692), 类别为0中各单词的出现概率
    :param pClass1: type=tensor, shape=(1,), 类别为1的概率
    :param pClass0: type=tensor, shape=(1,), 类别为0的概率

    :return result: type=tf.bool, shape=(1,), True代表分类结果为1，False代表分类结果为0
    """

    # 获取测试集数据个数
    m = testMat.get_shape().as_list()[0]

    # 对各类别中各单词的出现概率进行复制, shape=(m, 692)
    p1Vec = tf.tile(p1Vec, [m, 1])
    p0Vec = tf.tile(p0Vec, [m, 1])

    # 计算分类结果的概率, shape=(m, 1)
    p1 = tf.reduce_sum(tf.multiply(testMat, p1Vec), axis=1, keepdims=True) + tf.log(tf.tile(tf.expand_dims(pClass1, 1), [m, 1]))
    p0 = tf.reduce_sum(tf.multiply(testMat, p0Vec), axis=1, keepdims=True) + tf.log(tf.tile(tf.expand_dims(pClass0, 1), [m, 1]))

    # 取概率最大的类作为结果, shape=(m, 1)
    result = tf.greater(tf.subtract(p1, p0), tf.zeros([m, 1]))
    return result


if __name__ == '__main__':
    # 获取数据集
    trainMat, trainClass, testMat, testClass = loadDataSet()

    # tf.placeholder
    x_train = tf.placeholder(dtype=tf.float32, shape=trainMat.shape)
    y_train = tf.placeholder(dtype=tf.float32, shape=trainClass.shape)
    x_test = tf.placeholder(dtype=tf.float32, shape=testMat.shape)
    y_test = tf.placeholder(dtype=tf.float32, shape=testClass.shape)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        # 调用bayes()函数获得概率
        pOneWords, pZeroWords, pClassOne, pClassZero = bayes(x_train, y_train)

        # 调用classify()函数获得分类结果
        y = classify(x_test, pOneWords, pZeroWords, pClassOne, pClassZero)

        # 打印测试集的分类结果
        classifyResult = sess.run(y, feed_dict={x_train: trainMat, y_train: trainClass, x_test: testMat, y_test: testClass})
        print('Test set classify result: ')
        print(np.squeeze(np.array(classifyResult, dtype=int)))
