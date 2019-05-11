import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))  # 将所有数据转换为float类型
        dataMat.append(fltLine)
    return np.array(dataMat)


def randCent(dataSet, k):
    # 得到数据集的列数
    n = np.shape(dataSet)[1]
    # 得到一个K*N的空矩阵
    centroids = np.mat(np.zeros((k, n)))
    # 对于每一列
    for j in range(n):
        # 得到最小值
        minJ = min(dataSet[:, j])
        # 得到当前列的范围
        rangeJ = float(max(dataSet[:, j]) - minJ)
        # 在最小值和最大值之间取值
        centroids[:, j] = np.mat(minJ + rangeJ * np.random.rand(k, 1))
    return centroids


def figShow(dataSet, center, label, k):
    rect = [0.15, 0.1, 0.8, 0.8]
    fig = plt.figure()
    ax = fig.add_axes(rect, label='ax', frameon=False)
    plt.ylim((115.6, 117.2))
    for i in range(k):
        idx = np.where(np.array(label) == i)
        ax.scatter(dataSet[idx, 0], dataSet[idx, 1], marker='o')
    ax.scatter(center[:, 0], center[:, 1], marker='+', s=300)
    plt.show()


# dataSet (400, 2)
# center (k, 2)
def kmeans(dataSet, k, center):
    m = dataSet.get_shape().as_list()[0]

    # shape = (k, 400, 2)
    repDataSet = tf.tile(tf.expand_dims(dataSet, axis=0), [k, 1, 1])
    # shape = (k, 400, 2)
    repCenter = tf.tile(tf.expand_dims(center, axis=1), [1, m, 1])
    # shape = (k, 400, 1)
    distance = tf.sqrt(tf.reduce_sum(tf.pow(repCenter - repDataSet, 2), axis=2))
    # shape = (400)
    label = tf.squeeze(tf.argmin(distance, axis=0))

    # 计算簇内平均
    # shape = (k, 2)
    total = tf.unsorted_segment_sum(dataSet, label, k)
    # shape = (k, 2)
    num = tf.unsorted_segment_sum(tf.ones(dataSet.get_shape().as_list()), label, k)
    # shape = (k, 2)
    centerMean = total / num
    # shape = (1)
    change = tf.reduce_any(tf.not_equal(center, centerMean))
    return centerMean, label, change


dataSet = loadDataSet('Restaurant_Data_Beijing.txt')
k = 5

if __name__ == '__main__':
    x_input = tf.placeholder(dtype=tf.float32, shape=dataSet.shape)
    center = tf.Variable(randCent(dataSet, k), dtype=tf.float32)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        change_ = True
        while change_:
            center, label, change = kmeans(x_input, k, center)
            center_, label_, change_ = sess.run([center, label, change], feed_dict={x_input: dataSet})

    figShow(dataSet, center_, label_, k)
