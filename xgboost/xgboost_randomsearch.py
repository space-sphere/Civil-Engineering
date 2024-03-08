import numpy as np
import pandas as pd
import warnings
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
df = pd.read_csv(r"D:\program\pycharm\model\Data\out.csv")
X = df.iloc[:, 1:]
y = df.iloc[:, 0] - 1
from sklearn.model_selection import RandomizedSearchCV
clf = XGBClassifier()
r = RandomizedSearchCV(
    estimator=clf,
    param_distributions={
        'n_estimators': np.arange(20, 300, 10), # 弱分类器的个数
        'max_depth': np.arange(1, 20, 1),       # 弱分类器（CART回归树）的最大深度
        'min_child_weight': np.arange(1, 10, 1), # 分裂内部节点所需的最小样本数
        'learning_rate': np.arange(1e-3, 5e-1, 0.005),  # 学习率
        'gamma': np.arange(1e-3, 2e-1, 0.01),
        'colsample_bytree': np.arange(5e-1, 1, 0.01),
    },
    scoring='accuracy', n_iter=20, cv=5)
r.fit(X, y)
print('最终的准确率: {}'.format(r.best_score_))
print('所用参数')
print(r.best_params_)
params = r.best_params_
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
clf = XGBClassifier(**params)
train_and_evaluate(clf, X_train, X_test, y_train, y_test)
