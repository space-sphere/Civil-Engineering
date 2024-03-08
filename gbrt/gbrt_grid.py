import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv(r"D:\program\pycharm\model\Data\out.csv")
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
params = {
    'loss': 'deviance',
    'random_state': 1
}
clf = GradientBoostingClassifier(**params)
grid = GridSearchCV(
    estimator=clf,
    param_grid={
        'n_estimators': np.arange(100, 300, 10),
        'max_features': np.arange(1, 11, 1),
        'max_depth': np.arange(1, 11, 1),
        'learning_rate': np.arange(0.001, 0.5, 0.05),
        'min_samples_split': np.arange(1, 11, 1),
        'subsample': np.arange(0.8, 1, 0.01)
    },
    cv=5,
    scoring='accuracy',
    verbose=0, n_jobs=-1)
grid.fit(X, y)
print('最终的准确率: {}'.format(grid.best_score_))
print('所用参数')
params = grid.best_params_
print(params)
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
clf = GradientBoostingClassifier(**params)
train_and_evaluate(clf, X_train, X_test, y_train, y_test)
