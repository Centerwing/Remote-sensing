from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import pydot
import time
import numpy as np
import os

DATA_PATH = '../Dataset/300x300Data.npz'


def train(x_data, y_data):
    clf = DecisionTreeClassifier()

    clf.fit(x_data, y_data)
    save_model(clf)


def save_model(model):
    if 'model' not in os.listdir():
        os.makedirs('model')
    joblib.dump(model, 'model/nn.pkl')


def load_model():
    return joblib.load('model/nn.pkl')


def save_report(ytest, ypred, train_time, test_time):
    if 'report' not in os.listdir():
        os.makedirs('report')
    report_file = open('report/' + 'report.txt', 'a')
    report_file.seek(0)
    report_file.truncate()
    report = classification_report(ytest, ypred, labels=range(30))
    report_file.write(report + '\n')
    report_file.write('train_time = ' + str(train_time) + '\n')
    report_file.write('test_time = ' + str(test_time) + '\n')
    report_file.close()


def export():
    tree = load_model()
    export_graphviz(tree, out_file="tree.dot",
                    impurity=False,
                    filled=True)
    (graph,) = pydot.graph_from_dot_file('tree.dot')
    graph.write_png('tree.png')


def test(x_test, y_test, train_time):
    clf = load_model()

    st = time.clock()
    y_pred = clf.predict(x_test)
    et = time.clock()

    save_report(y_test, y_pred, train_time, et-st)


if __name__ == '__main__':
    gray_data = np.load(DATA_PATH)
    x_data = gray_data['x_train']
    y_data = gray_data['y_train']
    x_test = gray_data['x_test']
    y_test = gray_data['y_test']

    '''归一化数据'''
    scaler = StandardScaler()
    scaler.fit(x_data)
    x_data = scaler.transform(x_data)
    x_test = scaler.transform(x_test)

    st = time.clock()
    #train(x_data, y_data)
    et = time.clock()

    #test(x_test, y_test, et-st)
