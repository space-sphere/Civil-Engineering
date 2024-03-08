import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

warnings.filterwarnings('ignore')
df = pd.read_csv(r"D:\program\pycharm\model\Data\out.csv")
X = df.iloc[:, 1:]
y = df.iloc[:, 0]
clf = xgb.XGBClassifier()
grid = GridSearchCV(
    estimator=clf,
    param_grid={
        'n_estimators': np.arange(20, 300, 10),
        'max_features': np.arange(1, 11, 1),
        'max_depth': np.arange(1, 11, 1),
        'learning_rate': np.arange(0.001, 0.5, 0.05),
        'gamma': [0, 0.05, 0.06, 0.07, 0.08, 0.1, 0.3, 0.5, 0.7, 1]
        'min_child_weight': np.arange(1, 11, 1),
        'colsample_bytree': np.arange(0.8, 1, 0.01)
    },
    cv=5, scoring='accuracy', verbose=0, n_jobs=-1)
grid.fit(X, y)
print('最终的准确率: {}'.format(grid.best_score_))
print('所用参数')
params = grid.best_params_
print(grid.best_score_)

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
clf = xgb.XGBClassifier(**params)
train_and_evaluate(clf, X_train, X_test, y_train, y_test)
