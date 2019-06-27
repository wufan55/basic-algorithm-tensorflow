import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    """
    加载数据集

    :param fileName: type=string, 数据文件的url

    :return dataMat: type=np.array, 数据集
    """

    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))  # 将所有数据转换为float类型
        dataMat.append(fltLine)
    return np.array(dataMat)


def figShow(dataSet, center, label, k):
    """
    图表展示分类结果

    :param dataSet: type=np.array, 数据集
    :param center: type=np.array, 中心点
    :param label: type=np.array, 数据集标签
    :param k: type=int, 类数量

    :return:
    """
    rect = [0.15, 0.1, 0.8, 0.8]
    fig = plt.figure()
    ax = fig.add_axes(rect, label='ax', frameon=False)
    plt.ylim((115.6, 117.2))
    for i in range(k):
        idx = np.where(np.array(label) == i)
        ax.scatter(dataSet[idx, 0], dataSet[idx, 1], marker='o')
    ax.scatter(center[:, 0], center[:, 1], marker='+', s=300)
    plt.show()


def randCent(dataSet, k):
    """
    生成k个随机中心

    :param dataSet: type=np.array, 数据集
    :param k: type=int, 中心点数量

    :return centroids: type=np.mat, k个中心点
    """

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


# dataSet (400, 2)
# center (k, 2)
def kmeans(dataSet, k, center):
    """
    kmeans算法实现

    :param dataSet: type=np.array, shape=(400, 2), 数据集
    :param k: type=int, 类数量
    :param center: type=np.array, shape=(k, 2), k个初始中心点

    :return centerMean: type=tensor, shape=(k, 2), 更新后的中心点
    :return label: type=tensor, shape=(400,), 数据集标签
    :return change: type=tf.bool, shape=(1), 中心点是否有更新
    """

    # 获取数据点个数
    m = dataSet.get_shape().as_list()[0]

    # 对数据集和中心点进行维度扩展和复制, 使二者形状一致, 可以直接进行整体的运算

    # 对dataSet在axis=0上进行扩维, 然后复制一维k次
    # 初始shape=(400, 2), 操作后shape=(k, 400, 2)
    # 操作后的shape, 一维表示有k个相同的数据集, 二维表示每个数据集有400个数据点, 三维表示每个数据点有2个值
    repDataSet = tf.tile(tf.expand_dims(dataSet, axis=0), [k, 1, 1])

    # 对center在axis=1上进行扩维, 然后复制二维m次
    # 初始shape=(k, 2), 操作后shape=(k, 400, 2)
    # 操作后的shape, 一维表示有k个不同的中心点, 二维表示每一个中心点有400个副本, 三维表示每个中心点有两个值
    repCenter = tf.tile(tf.expand_dims(center, axis=1), [1, m, 1])

    # 进行欧式距离计算
    # 得到distance, shape=(k, 400, 1)
    distance = tf.sqrt(tf.reduce_sum(tf.pow(repCenter - repDataSet, 2), axis=2))

    # 根据欧式距离最小获取标签
    # 对distance, shape=(k, 400, 1), 调用tf.argmin获取distance沿一维获取最小值的indices
    # 操作后shape=(400, 1), 一维表示400个数据点, 二维表示每个数据点的类标签
    # tf.squeeze对维度进行压缩, 压缩后shape=(400,), 得到label
    label = tf.squeeze(tf.argmin(distance, axis=0))

    # 对中心点进行更新

    # 统计各类的数据点的数值之和
    # 得到total, shape=(k, 2), 一维表示有k个类, 二维有两个值, 分别表示各类数据点的X和Y坐标之和
    total = tf.unsorted_segment_sum(dataSet, label, k)

    # 统计各类的数据点的数量
    # 得到num, shape=(k, 2), 一维表示有k个类, 二维有两个相等的值, 表示各类数据点的数量
    num = tf.unsorted_segment_sum(tf.ones(dataSet.get_shape().as_list()), label, k)

    # 计算簇内平均
    # 得到centerMean, shape=(k, 2)
    centerMean = total / num

    # 检查中心点是否不变
    change = tf.reduce_any(tf.not_equal(center, centerMean))
    return centerMean, label, change


if __name__ == '__main__':
    # constant
    dataSet = loadDataSet('Restaurant_Data_Beijing.txt')
    k = 5

    # tf.placeholder
    x_input = tf.placeholder(dtype=tf.float32, shape=dataSet.shape)

    # random center
    center = tf.Variable(randCent(dataSet, k), dtype=tf.float32)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        # 初始change为True, 当调用kmeans()返回的change为False跳出循环
        change_ = True
        while change_:
            center, label, change = kmeans(x_input, k, center)
            center_, label_, change_ = sess.run([center, label, change], feed_dict={x_input: dataSet})

    # 图表显示
    figShow(dataSet, center_, label_, k)
