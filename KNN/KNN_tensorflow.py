import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# 创建简单数据集
def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = np.array([0, 0, 1, 1])
    return group, labels


# k近邻算法
def classify(inX, dataSet, labels, k):
    # inX用于分类的输入向量, tensor, shape=(1, 2)
    # dataSet输入的训练样本集, tensor, shape=(m, 2)
    # labels为标签向量, tensor shape=(m,)
    # k用于选择最近的邻居数目

    # 计算距离
    m = dataSet.get_shape().as_list()[0]

    # shape = (m, 1)
    distances = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf.tile(inX, (m, 1)), dataSet)), axis=1))

    sortedDistIndicies = tf.nn.top_k(-distances, k).indices

    ids = tf.cast(tf.gather(labels, sortedDistIndicies), dtype=tf.int32)
    classNum = tf.size(tf.unique(ids)[0])
    num = tf.unsorted_segment_sum(tf.ones(k), ids, classNum)

    result = tf.argmax(num)
    return result


def figShow(inX, result, dataSet, label):
    fig = plt.figure()
    rec = [0.15, 0.15, 0.8, 0.8]
    ax = fig.add_axes(rec, label='ax', frameon=False)
    for i in range(len(set(label))):
        idx = np.where(np.array(label) == i)
        ax.scatter(dataSet[idx, 0], dataSet[idx, 1], marker='o', label=i)
    ax.scatter(inX[0][0], inX[0][1], marker='+', s=200, label=result)
    plt.legend(loc='best')
    plt.show()


def knn(k=3):
    dataSet, label = createDataSet()
    point = [[0.3, 0.2]]

    x_ = tf.placeholder(dtype=tf.float32, shape=(1, dataSet.shape[1]))
    y_ = tf.placeholder(dtype=tf.float32, shape=label.shape)
    data = tf.placeholder(dtype=tf.float32, shape=dataSet.shape)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        result = classify(x_, data, y_, k)
        result = sess.run(result, feed_dict={x_: point, y_: label, data: dataSet})

    figShow(point, result, dataSet, label)


if __name__ == '__main__':
    knn(k=3)
