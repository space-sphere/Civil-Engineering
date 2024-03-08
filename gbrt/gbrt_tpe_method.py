from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import warnings
import hyperopt

warnings.filterwarnings('ignore')
df = pd.read_csv(r"D:\program\pycharm\model\Data\out.csv")
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

def hyperopt_object(params):
    clf = GradientBoostingClassifier(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        min_samples_split=int(params['min_samples_split']),
        learning_rate=params['learning_rate'],
        max_features=int(params['max_features']),
        loss=params['loss'],
        subsample=params['subsample'],
        random_state=params['random_state']
    )
    scores = np.mean(cross_val_score(clf, X, y, scoring='accuracy', cv=5))
    return 1 - scores
params = {
    'n_estimators': hyperopt.hp.quniform("n_estimators", 100, 300, 10), # 弱分类器的个数
    'max_depth': hyperopt.hp.randint('max_depth', 1, 15),       # 弱分类器（CART回归树）的最大深度
    'min_samples_split': hyperopt.hp.randint('min_samples_split', 1, 15), # 分裂内部节点所需的最小样本数
    'learning_rate': hyperopt.hp.uniform('learning_rate', 1e-3, 5e-1),  # 学习率
    'max_features': hyperopt.hp.randint('max_features', 1, 10),
    'loss': 'deviance',
    'subsample': hyperopt.hp.uniform('subsample', 5e-1, 1),
    'random_state': 1
}
trials = hyperopt.Trials()
res = hyperopt.fmin(
    hyperopt_object,
    space=params,
    algo=hyperopt.tpe.suggest,
    max_evals=20,
    trials=trials)
res['n_estimators'] = int(res['n_estimators'])
res['max_depth'] = int(res['max_depth'])
res['min_samples_split'] = int(res['min_samples_split'])
res['max_features'] = int(res['max_features'])
print('所用参数')
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
    # train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train, train_sizes=np.linspace(0.01, 1.0, 100, ), cv=5)

    print ("训练集精度:")   # 训练集精度
    print (clf.score(X_train, y_train))

    print ("\n测试集精度:")    # 测试集精度
    print (clf.score(X_test, y_test))

    y_pred = clf.predict(X_test)

    print ("\n分类识别报告:")      # 分类识别报告
    print (metrics.classification_report(y_test, y_pred))
    # return train_sizes, train_scores, test_scores
res = {'learning_rate': 0.09278943037570718, 'max_depth': 6, 'max_features': 1, 'min_samples_split': 13, 'n_estimators': 270, 'subsample': 0.95866409482895}
clf = GradientBoostingClassifier(**res)
train_and_evaluate(clf, X_train, X_test, y_train, y_test)
