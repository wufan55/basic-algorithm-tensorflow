import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.contrib.tensor_forest.client import random_forest


# 加载数据集并且进行shuffle操作
def createDataSet():
    iris = datasets.load_iris()
    index = np.random.choice(150, 150, replace=False)
    x = np.array(iris.data[:, [0, 2]])[index]
    y = np.array(iris.target)[index]
    return x.astype(np.float32), y.astype(np.int)


def figShowAndWrite(dataSet, label):
    featureNum = dataSet.shape[1]
    classNum = len(set(label))

    params = tensor_forest.ForestHParams(num_classes=classNum, num_features=featureNum, num_trees=1, max_nodes=20)
    classifier = random_forest.TensorForestEstimator(params)
    classifier.fit(dataSet, label)

    # 画图
    x_min, x_max = dataSet[:, 0].min()-1, dataSet[:, 0].max()+1
    y_min, y_max = dataSet[:, 1].min()-1, dataSet[:, 1].max()+1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    Z = np.array(list(classifier.predict(np.c_[xx.ravel(), yy.ravel()].astype(np.float32))))
    for i in range(len(Z)):
        Z[i] = Z[i]['classes']
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.2)
    plt.scatter(dataSet[:, 0], dataSet[:, 1], c=label+3, alpha=1)
    plt.show()


if __name__ == '__main__':
    dataSet, label = createDataSet()
    figShowAndWrite(dataSet, label)
