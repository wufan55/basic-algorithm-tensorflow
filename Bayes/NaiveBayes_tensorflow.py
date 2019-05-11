import tensorflow as tf
import numpy as np


def loadDataSet():
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


# trainMat = shape(40, 692)
# trainClass = shape(40,)
# testMat = shape(10, 692)
# testClass = shape(10,)
def bayes(trainMat, trainClass):
    numTrain = trainMat.get_shape().as_list()[0]
    numWords = trainMat.get_shape().as_list()[1]

    pClassOne = tf.divide(tf.reduce_sum(trainClass, keepdims=True), tf.cast(numTrain, dtype=tf.float32))
    pClassZero = 1 - pClassOne

    # shape = (1, 692)
    pOneWordsNum = tf.reduce_sum(tf.multiply(trainMat, tf.tile(tf.expand_dims(trainClass, 1), [1, numWords])), axis=0, keepdims=True)
    pZeroWordsNum = tf.reduce_sum(tf.multiply(trainMat, tf.tile(tf.expand_dims(1-trainClass, 1), [1, numWords])), axis=0, keepdims=True)

    # shpae = (1, 1)
    pOneWordsTotal = tf.reduce_sum(pOneWordsNum, axis=1, keepdims=True)
    pZeroWordsTotal = tf.reduce_sum(pZeroWordsNum, axis=1, keepdims=True)

    # shape = (1, 692)
    pOneWords = tf.divide(pOneWordsNum, tf.tile(pOneWordsTotal, [1, numWords]))
    pZeroWords = tf.divide(pZeroWordsNum, tf.tile(pZeroWordsTotal, [1, numWords]))

    return pOneWords, pZeroWords, pClassOne, pClassZero


# testMat shape(10, 692)
# p1Vec shape(1, 692)
# pClass1 shape(1,)
# return True 类别为1，False 类别为0
def classify(testMat, p1Vec, p0Vec, pClass1, pClass0):
    m = testMat.get_shape().as_list()[0]

    # shape = (10, 692)
    p1Vec = tf.tile(p1Vec, [m, 1])
    p0Vec = tf.tile(p0Vec, [m, 1])

    # shape = (10, 1)
    p1 = tf.reduce_sum(tf.multiply(testMat, p1Vec), axis=1, keepdims=True) + tf.log(tf.tile(tf.expand_dims(pClass1, 1), [m, 1]))
    p0 = tf.reduce_sum(tf.multiply(testMat, p0Vec), axis=1, keepdims=True) + tf.log(tf.tile(tf.expand_dims(pClass0, 1), [m, 1]))

    # shape = (10, 1)
    return tf.greater(tf.subtract(p1, p0), tf.zeros([m, 1]))


if __name__ == '__main__':
    trainMat, trainClass, testMat, testClass = loadDataSet()

    x_train = tf.placeholder(dtype=tf.float32, shape=trainMat.shape)
    y_train = tf.placeholder(dtype=tf.float32, shape=trainClass.shape)
    x_test = tf.placeholder(dtype=tf.float32, shape=testMat.shape)
    y_test = tf.placeholder(dtype=tf.float32, shape=testClass.shape)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        pOneWords, pZeroWords, pClassOne, pClassZero = bayes(x_train, y_train)
        y = classify(x_test, pOneWords, pZeroWords, pClassOne, pClassZero)

        # 打印测试集的分类结果
        classifyResult = sess.run(y, feed_dict={x_train: trainMat, y_train: trainClass, x_test: testMat, y_test: testClass})
        print(np.squeeze(np.array(classifyResult, dtype=int)))
