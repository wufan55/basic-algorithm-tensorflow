import tensorflow as tf
import numpy as np
import itertools


def createDataSet():
    """
    创建简单数据集

    :return dataSet: type=list, 数据集
    """

    dataSet = [[1, 3, 4],
               [2, 3, 5],
               [1, 2, 3, 5],
               [2, 5]]
    return dataSet


def getTotal(dataSet, items):
    """
    获取items在dataSet中的出现次数

    :param dataSet: type=list, 数据集
    :param items: type=list或int, 被统计的元素

    :return total: type=list, items中各item的出现次数
    """

    def getTime(dataSet, item):
        """
        获取item在dataSet中的出现次数

        :param dataSet: type=list, 数据集
        :param item: type=list或int, 被统计的元素

        :return time: type=int, 出现次数
        """

        time = 0

        # 如果item为int, 则转换成list
        if isinstance(item, int):
            item = [item]

        # 统计出现次数
        for i in range(len(dataSet)):
            if np.array_equal(np.array(item), list(set(dataSet[i]) & set(np.array(item)))):
                time += 1
        return time

    # 遍历items, 对其中元素逐一调用getTime()方法统计出现次数
    total = []
    for i in range(len(items)):
        total.append(getTime(dataSet, items[i]))
    return total


def getItems(target):
    """
    对target中的元素进行C2排列组合

    :param target: type=list, 进行排列组合的list

    :return result: type=list, C2排列组合后的结果
    """

    # 取target中两个元素进行排列组合
    a = itertools.combinations(target, 2)

    item = np.array([x for x in a])
    result = []
    for i in range(len(item)):
        result.append(list(set(np.ravel(item[i]))))

    if len(np.array(result).shape) == 1:
        return np.unique(np.array(result))
    return np.unique(np.array(result), axis=0)


def getDict(dataSet):
    """
    获取dataSet的字典

    :param dataSet: type=list, 数据集

    :return dic: type=list, 数据集的字典
    """

    dic = []
    for i in range(len(dataSet)):
        dic = list(set(dataSet[i]) | set(dic))
    return dic


def apriori(minSupport=2):
    """
    Apriori算法实现

    :param minSupport: type=int, 最小支持度

    :return delItem: type=list, 大于最小支持度的规则
    """

    with tf.Session() as sess:
        # 创建数据集
        dataSet = createDataSet()

        # 获取数据集的字典
        item = getDict(dataSet)

        # 定义一个储存中间过程值的变量
        temp = []
        while True:
            # 统计每个元素在dataSet中出现次数
            total = getTotal(dataSet, item)

            # 获取个数小于minSupport的索引
            index = sess.run(tf.squeeze(tf.where(tf.less(tf.constant(total), minSupport))))

            # 删除个数小于minSupport的元素
            delItem = np.delete(np.array(item), index, axis=0)

            # 当只剩下一条规则, 不进行接下来的排列组合操作, 跳出循环
            if len(delItem) == 1:
                return delItem

            # 当所有规则都小于minSupport时, 跳出循环返回上一个循环生成的规则
            if len(delItem) == 0:
                return temp

            # 记录当前循环符合要求的规则
            temp = delItem

            # 进行排列组合
            item = getItems(delItem)


if __name__ == '__main__':
    print('minSupport = 2 : ')
    print(apriori(2))
    print('minSupport = 3 : ')
    print(apriori(3))
    print('minSupport = 4 : ')
    print(apriori(4))
