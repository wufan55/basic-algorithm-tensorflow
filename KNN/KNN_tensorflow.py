import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def createDataSet():
    """
    创建简单数据集

    :return group: type=np.array, shape=(4, 2), 表示数据集, shape的一维表示数据点数量, 二维表示数据点包含多少个值
    :return labels: type=np.array, shape=(4,), 表示数据集对应的标签
    """

    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = np.array([0, 0, 1, 1])
    return group, labels


def figShow(inX, result, dataSet, label):
    """
    图表展示分类结果

    :param inX: type=np.array, shape=(1, 2), 表示已分类的数据点
    :param result: type=int, 表示inX的分类结果
    :param dataSet: type=np.array, shape=(4, 2), 分类基于的数据集
    :param label: type=np.array, shape=(1, 2), 分类基于的数据集的标签

    :return:
    """

    fig = plt.figure()
    rec = [0.15, 0.15, 0.8, 0.8]
    ax = fig.add_axes(rec, label='ax', frameon=False)
    for i in range(len(set(label))):
        idx = np.where(np.array(label) == i)
        ax.scatter(dataSet[idx, 0], dataSet[idx, 1], marker='o', label=i)
    ax.scatter(inX[0][0], inX[0][1], marker='+', s=200, label=result)
    plt.legend(loc='best')
    plt.show()


def classify(inX, dataSet, labels, k):
    """
    基于knn进行分类

    :param inX: type=tensor, shape=(1, 2), 表示待分类的数据点
    :param dataSet: type=tensor, shape=(m, 2), 分类基于的数据集
    :param labels: type=tensor, shape=(m,), 分类基于的数据集的标签
    :param k: type=int, 表示用于选择最近的邻居数目

    :return result: type=tf.int, 表示inX的分类结果
    """

    # 获取dataSet的数据点个数
    m = dataSet.get_shape().as_list()[0]

    # 计算inX与数据集中各数据点的距离
    # 先对inX进行复制, 在一维上复制m次使其的shape与dataSet的shape相同
    # inX与dataSet计算欧式距离
    # 得到distances, shape=(m, 1)
    distances = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf.tile(inX, (m, 1)), dataSet)), axis=1))

    # 获取distances中最小项的indices
    # tf.nn.top_k函数获取最大的k项, 使用时先把distances取反
    # 得到sortedDistIndicies, shape=(3,)
    sortedDistIndicies = tf.nn.top_k(-distances, k).indices

    # 根据sortedDistIndicies获取dataSet中距离inX最小的k项的标签
    ids = tf.cast(tf.gather(labels, sortedDistIndicies), dtype=tf.int32)

    # 统计k项中各个类的数据点数量
    classNum = tf.size(tf.unique(ids)[0])
    num = tf.unsorted_segment_sum(tf.ones(k), ids, classNum)

    # 取数据量最多的类作为分类结果
    result = tf.argmax(num)
    return result


if __name__ == '__main__':
    # 获取数据集, 标签, 待分类的数据点
    dataSet, label = createDataSet()
    point = [[0.3, 0.2]]

    # tf.placeholder
    x_ = tf.placeholder(dtype=tf.float32, shape=(1, dataSet.shape[1]))
    y_ = tf.placeholder(dtype=tf.float32, shape=label.shape)
    data = tf.placeholder(dtype=tf.float32, shape=dataSet.shape)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        # 调用classify函数获得结果
        result = sess.run(classify(x_, data, y_, 3), feed_dict={x_: point, y_: label, data: dataSet})

    # 图表显示
    figShow(point, result, dataSet, label)
