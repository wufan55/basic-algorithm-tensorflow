import tensorflow as tf
import numpy as np


class BinaryTree:
    def __init__(self, rootObj):
        self.key = rootObj
        self.leftChild = None
        self.rightChild = None

    def insertLeft(self, newNode):
        if self.leftChild == None:
            self.leftChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.leftChild = self.leftChild
            self.leftChild = t

    def insertRight(self, newNode):
        if self.rightChild == None:
            self.rightChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.rightChild = self.rightChild
            self.rightChild = t


# simple dataset
#     不浮出水面是否可以生存 是否有脚蹼  是否属于鱼类
# 1           是               是          是
# 2           是               是          是
# 3           是               否          否
# 4           否               是          否
# 5           否               是          否
def createDataSet():
    dataSet = [[1, 1],
               [1, 1],
               [1, 0],
               [0, 1],
               [0, 1]]
    label = [1, 1, 0, 0, 0]
    return np.array(dataSet, dtype=float), np.array(label, dtype=float)


def probabilityCal(dataSet, label):
    total = tf.shape(label)
    #
    idZero = tf.squeeze(tf.where(tf.equal(label, 0)))
    idOne = tf.squeeze(tf.where(tf.equal(label, 1)))

    zeroNum = tf.unsorted_segment_sum(tf.ones(total), idZero, 2)[0]
    oneNum = tf.unsorted_segment_sum(tf.ones(total), idOne, 2)[0]

    zeroP = tf.divide(zeroNum, tf.cast(total, dtype=tf.float32))
    oneP = tf.divide(oneNum, tf.cast(total, dtype=tf.float32))
    return zeroP, oneP


# dataSet (m, 2)
# label (m,)
def entropyCal(dataSet, label, index=None):
    # dataSet = shape(dataNum, labelNum)
    def splitDataSet(dataSet, label, index):
        t = dataSet[:, index]
        idZero = tf.squeeze(tf.where(tf.equal(t, 0)))
        idOne = tf.squeeze(tf.where(tf.equal(t, 1)))

        return tf.gather(dataSet, idZero), tf.gather(dataSet, idOne), tf.gather(label, idZero), tf.gather(label, idOne)

    total = tf.shape(label)
    if index is None:
        zeroP, oneP = probabilityCal(dataSet, label)
        return -tf.multiply(zeroP, tf.log(zeroP))-tf.multiply(oneP, tf.log(oneP))
    else:
        m = index.shape[0]
        for i in range(m):
            dataSet_1, dataSet_2, label_1, label_2 = splitDataSet(dataSet, label, index[i])

            zeroP, oneP = probabilityCal(dataSet_1, label_1)
            e1 = tf.cast(tf.shape(label_1), dtype=tf.float32) / tf.cast(total, dtype=tf.float32)
            entropyZero = -tf.multiply(zeroP, tf.log(zeroP))-tf.multiply(oneP, tf.log(oneP))

            zeroP, oneP = probabilityCal(dataSet_2, label_2)
            e2 = tf.cast(tf.shape(label_2), dtype=tf.float32) / tf.cast(total, dtype=tf.float32)
            entropyOne = -tf.multiply(zeroP, tf.log(zeroP))-tf.multiply(oneP, tf.log(oneP))

            tf.assign(entropy[i], e1 * entropyZero + e2 * entropyOne)
        return entropy


def getBestPoint(dataSet, label):
    m = dataSet.get_shape().as_list()[1]
    # shape(m, 1)
    dataSetEntropy = tf.tile(tf.expand_dims(entropyCal(dataSet, label), 0), [m, 1])

    index = tf.constant(np.array(range(m)))
    # shape(m, 1)
    splitDataEntropy = entropyCal(dataSet, label, index)
    entropyDiff = dataSetEntropy - splitDataEntropy

    return tf.argmax(entropyDiff)


def dataSetSplit(dataSet, label, index):
    m = dataSet.get_shape().as_list()[1]
    idZero = tf.where(tf.equal(dataSet[:, index], tf.zeros(1)))
    idOne = tf.where(tf.equal(dataSet[:, index], tf.ones(1)))
    dataSet = tf.concat(1, [dataSet[:, [x for x in range(0, index)]], dataSet[:, [x for x in range(index+1, m)]]])
    return dataSet[idZero], dataSet[idOne], label[idZero], label[idOne]


if __name__ == '__main__':
    dataSet, label = createDataSet()

    x_input = tf.placeholder(tf.float32, dataSet.shape)
    y_input = tf.placeholder(tf.float32, label.shape)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        index = sess.run(getBestPoint(x_input, y_input), feed_dict={x_input: dataSet, y_input: label})
        root = BinaryTree(index)

        dataSet_1, dataSet_2, label_1, label_2 = dataSetSplit(x_input, y_input, index)
        left = sess.run(getBestPoint(dataSet_1, label_1), feed_dict={x_input: dataSet, y_input: label})
        right = sess.run(getBestPoint(dataSet_2, label_2), feed_dict={x_input: dataSet, y_input: label})
        root.insertLeft(BinaryTree(left))
        root.insertRight(BinaryTree(right))

        print(root.key, root.leftChild, root.rightChild)
