import tensorflow as tf
import numpy as np
import itertools


def createDataSet():
    dataSet = [[1, 3, 4],
               [2, 3, 5],
               [1, 2, 3, 5],
               [2, 5]]
    return dataSet


def getTotal(dataSet, item):
    def getTime(dataSet, item):
        time = 0
        if isinstance(item, int):
            item = [item]
        for i in range(len(dataSet)):
            if np.array_equal(np.array(item), list(set(dataSet[i]) & set(np.array(item)))):
                time += 1
        return time

    total = []
    for i in range(len(item)):
        total.append(getTime(dataSet, item[i]))
    return total


def getItems(target):
    a = itertools.combinations(target, 2)
    item = np.array([x for x in a])
    temp = []
    for i in range(len(item)):
        temp.append(list(set(np.ravel(item[i]))))

    if len(np.array(temp).shape) == 1:
        return np.unique(np.array(temp))
    return np.unique(np.array(temp), axis=0)


def getDict(dataSet):
    # 获取dataSet中元素
    item = []
    for i in range(len(dataSet)):
        item = list(set(dataSet[i]) | set(item))
    return item


def apriori(minSupport=2):
    with tf.Session() as sess:
        # 创建数据集
        dataSet = createDataSet()

        # 获取数据集的字典
        item = getDict(dataSet)

        while True:
            # 统计每个元素在dataSet中出现次数
            total = getTotal(dataSet, item)

            # 获取个数小于minSupport的索引
            index = sess.run(tf.squeeze(tf.where(tf.less(tf.constant(total), minSupport))))

            # 删除个数小于minSupport的元素
            delItem = np.delete(np.array(item), index, axis=0)

            # 进行排列组合
            item = getItems(delItem)

            if len(item) == 0:
                return delItem


if __name__ == '__main__':
    print(apriori(2))
