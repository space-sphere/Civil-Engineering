import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import warnings
import hyperopt
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore')
df = pd.read_csv(r"../../../Data/隧道围岩项目/训练集.csv")
# X = df.iloc[:, 1:]
# y = df.iloc[:, 0]
# data = data_normalize(d, 'WD')
# df = data.fillna_by_random_fix()
X = df.iloc[:, 1:]
y = df.iloc[:, 0]
from sklearn.svm import SVC
def hyperopt_object(params):
    clf = SVC(
        C=params['C'],
        kernel=params['kernel'],
        gamma=params['gamma'],
        random_state=0
    )
    scores = np.mean(cross_val_score(clf, X, y, scoring='accuracy', cv=5))
    return 1 - scores
params = {
    'gamma': hyperopt.hp.loguniform('gamma', np.log(0.001), np.log(0.1)),
    'kernel':hyperopt.hp.choice('kernel',['rbf','poly']),
    'C':hyperopt.hp.loguniform("C", np.log(1), np.log(100))
}
trials = hyperopt.Trials()
res = hyperopt.fmin(
    hyperopt_object,
    space=params,
    algo=hyperopt.tpe.suggest,
    max_evals=20,
    trials=trials)
res['kernel'] = ['rbf','poly'][res['kernel']]
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
clf = SVC(**res)
train_and_evaluate(clf, X_train, X_test, y_train, y_test)
