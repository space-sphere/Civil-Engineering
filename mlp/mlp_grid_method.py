import numpy as np
import pandas as pd
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')
df = pd.read_csv(r"D:\program\pycharm\model\Data\out.csv")
X = df.iloc[:, 1:]
y = df.iloc[:, 0]
count = 0
clf = MLPClassifier(max_iter=1000, random_state=2)
grid = GridSearchCV(
    estimator=clf,
    param_grid={
        'alpha': np.arange(1e-5, 1e-2, 3e-6),
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'hidden_layer_sizes': [(100, 100), (150, 100), (200, 150), (100, )],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'momentum': np.linspace(0, 1, 20),
        'activation': ['relu', 'tanh'],
        'learning_rate_init': np.arange(1e-3, 5e-1, 0.005)
    },
    cv=5, scoring='accuracy', verbose=0, n_jobs=-1)
grid.fit(X, y)
print('最终的准确率: {}'.format(grid.best_score_))
print('所用参数')
params = grid.best_params_
print(grid.best_params_)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
aver = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train-aver)/std
X_test = (X_test-aver)/std
from sklearn import metrics
def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)             # 训练

    print ("训练集精度:")   # 训练集精度
    print (clf.score(X_train, y_train))

    print ("\n测试集精度:")    # 测试集精度
    print (clf.score(X_test, y_test))

    y_pred = clf.predict(X_test)

    print ("\n分类识别报告:")      # 分类识别报告
    print (metrics.classification_report(y_test, y_pred))
clf = MLPClassifier(**params)
train_and_evaluate(clf, X_train, X_test, y_train, y_test)

