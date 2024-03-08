import numpy as np
import pandas as pd
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import hyperopt

warnings.filterwarnings('ignore')
df = pd.read_csv(r"D:\program\pycharm\model\Data\out.csv")
X = df.iloc[:, 1:]
y = df.iloc[:, 0]
def hyperopt_object(params):
    clf = MLPClassifier(
        random_state=2,
        max_iter=1000,
        **params
    )
    scores = np.mean(cross_val_score(clf, X, y, scoring='accuracy', cv=5))
    return 1 - scores
params = {
    'alpha': hyperopt.hp.uniform('alpha', 1e-5, 1e-2),
    'hidden_layer_sizes': hyperopt.hp.choice('hidden_layer_sizes', [(50, 50, 50, 50, 50), (100, 100), (150, 100), (200, 150), (100, )]),
    'learning_rate': hyperopt.hp.choice('learning_rate', ['constant', 'invscaling', 'adaptive']),
    'solver': hyperopt.hp.choice('solver', ['adam', 'sgd', 'lbfgs']),
    'momentum': hyperopt.hp.uniform('momentum', 0, 1),
    'activation': hyperopt.hp.choice('activation', ['tanh', 'relu']),
    'learning_rate_init': hyperopt.hp.uniform('learning_rate_init', 1e-3, 5e-1)
}
trials = hyperopt.Trials()
res = hyperopt.fmin(
    hyperopt_object,
    space=params,
    algo=hyperopt.tpe.suggest,
    max_evals=20,
    trials=trials)
print('所用参数')
res['hidden_layer_sizes'] = [(100, 100), (150, 100), (200, 150), (100, )][res['hidden_layer_sizes']]
res['activation'] = ['tanh', 'relu'][res['activation']]
res['learning_rate'] = ['constant', 'invscaling', 'adaptive'][res['learning_rate']]
res['solver'] = ['adam', 'sgd', 'lbfgs'][res['solver']]
print(res)

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
clf = MLPClassifier(**res)
train_and_evaluate(clf, X_train, X_test, y_train, y_test)
