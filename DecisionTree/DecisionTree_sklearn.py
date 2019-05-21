import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


# 加载数据集并且进行shuffle操作
def createDataSet():
    iris = datasets.load_iris()
    index = np.random.choice(150, 150, replace=False)
    x = np.array(iris.data[:, [0, 2]])[index]
    y = np.array(iris.target)[index]
    return x, y


def createTree(dataSet, label):
    DT = DecisionTreeClassifier(criterion='gini', max_depth=4)
    DT.fit(dataSet, label)
    return DT


def figShow(dataSet, label):
    DT = createTree(dataSet, label)
    x_min, x_max = dataSet[:, 0].min()-1, dataSet[:, 0].max()+1
    y_min, y_max = dataSet[:, 1].min()-1, dataSet[:, 1].max()+1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    Z = DT.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.2)
    plt.scatter(dataSet[:, 0], dataSet[:, 1], c=label+3, alpha=1)
    plt.show()


if __name__ == '__main__':
    dataSet, label = createDataSet()
    figShow(dataSet, label)
