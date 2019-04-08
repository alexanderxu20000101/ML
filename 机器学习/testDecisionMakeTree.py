#用sklearn 中的对决策树的包装的包,直接用. 代码比较少.


from sklearn.model_selection import train_test_split
from pandas import DataFrame as df
from math import log
import pandas as pd
import operator
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.externals.six import StringIO
from sklearn import  tree  # 【tree 有两大函数 1、DecisionTreeClassifier--决策树构建
            # 2、DecisionTreeRegressor --回归决策树 3、 export_graphviz --决策树可视化
#           参数参考https://blog.csdn.net/ling_mochen/article/details/80011263  】
import pydotplus


def createDataSet():
    data = [[0, 0, 0, 0, 'no'],  # 数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    y_label = "isOk"
    x_labels = ["age", "work", "house", "credit"]
    return data, x_labels, y_label


if __name__ == '__main__':
    data, x_labels, y_label = createDataSet()
    x, y = [], []
    for i in data:
        y.append(i[-1])
        x.append(i[:-1])
    train_x = df(x, columns=x_labels)
    train_y = df(y, columns=[y_label])
    clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)  # 创建DecisionTreeClassifier()类
    clf = clf.fit(x, y)  # 使用数据，构建决策树
    reg_dot_data = tree.export_graphviz(clf, out_file=None,
                                        feature_names=train_x.keys(),
                                        class_names=clf.classes_)  # 决策树可视化函数
    reg_graph = pydotplus.graph_from_dot_data(reg_dot_data)
    reg_graph.write_png('tree.png')  # 保存为图片


