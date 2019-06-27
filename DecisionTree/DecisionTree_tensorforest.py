import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.contrib.tensor_forest.client import random_forest


def createDataSet():
    """
    加载iris数据集

    :return x: type=np.array, 数据集
    :return y: type=np.array, 标签
    """

    iris = datasets.load_iris()
    index = np.random.choice(150, 150, replace=False)
    x = np.array(iris.data[:, [0, 2]])[index]
    y = np.array(iris.target)[index]
    return x.astype(np.float32), y.astype(np.int)


def figShowAndWrite(dataSet, label):
    """
    对数据集进行决策树分类并用图表显示

    :param dataSet: 数据集
    :param label: 标签

    :return:
    """

    # 获取特征个数和类个数
    featureNum = dataSet.shape[1]
    classNum = len(set(label))

    # 调用高层api实现决策树

    # 根据参数生成type=ForestHParams的决策树参数
    params = tensor_forest.ForestHParams(num_classes=classNum, num_features=featureNum, num_trees=1, max_nodes=20)

    # 使用type=ForestHParams的参数生成决策树
    classifier = random_forest.TensorForestEstimator(params)

    # 决策树拟合训练集
    classifier.fit(dataSet, label)

    # 显示决策树的分类结果

    # 画图
    x_min, x_max = dataSet[:, 0].min()-1, dataSet[:, 0].max()+1
    y_min, y_max = dataSet[:, 1].min()-1, dataSet[:, 1].max()+1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    # 使用生成的决策树进行分类
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()].astype(np.float32))

    Z = np.array(list(Z))
    for i in range(len(Z)):
        Z[i] = Z[i]['classes']
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.2)
    plt.scatter(dataSet[:, 0], dataSet[:, 1], c=label+3, alpha=1)
    plt.show()


if __name__ == '__main__':
    dataSet, label = createDataSet()
    figShowAndWrite(dataSet, label)
