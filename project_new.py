import numpy as np
import pandas as pd
from data_pre import data_normalize
import warnings

d = pd.read_excel(r"D:\program\pycharm\model\Data\土木预测训练.xlsx", sheet_name='Sheet1')
data = data_normalize(d, 'WD')
df = data.fillna_by_random()
X = df.iloc[:, 1:]
y = df.iloc[:, 0] - 1
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
warnings.filterwarnings('ignore')
mlp = MLPClassifier()
scoresmlp = cross_val_score(mlp, X, y, scoring='accuracy', cv=5)
print('mlp模型在训练集和验证集上的准确率为：{}'.format(np.mean(scoresmlp)))
from xgboost import XGBClassifier
xgb = XGBClassifier()
scoresxgb = cross_val_score(xgb, X, y, scoring='accuracy', cv=5)
print('xgb模型在训练集和验证集上的准确率为：{}'.format(np.mean(scoresxgb)))
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
scoresgbc = cross_val_score(gbc, X, y, scoring='accuracy', cv=5)
print('gbrt模型在训练集和验证集上的准确率为：{}'.format(np.mean(scoresgbc)))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
scoresrfc = cross_val_score(rfc, X, y, scoring='accuracy', cv=5)
print('randomforest模型在训练集和验证集上的准确率为：{}'.format(np.mean(scoresrfc)))
print('加入测试集后的结果：')
def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)             # 训练

    print ("训练集精度:")   # 训练集精度
    print (clf.score(X_train, y_train))

    print ("\n测试集精度:")    # 测试集精度
    print (clf.score(X_test, y_test))

    y_pred = clf.predict(X_test)

    print ("\n分类识别报告:")      # 分类识别报告
    print (metrics.classification_report(y_test, y_pred))

    # print ("\n混淆矩阵:")
    # print (metrics.confusion_matrix(y_test, y_pred))
d_train = pd.read_excel('D:\program\pycharm\model\Data\土木训练测试.xlsx', sheet_name='Sheet1')
X_test = d_train.dropna(axis=0, how='any').iloc[:, 1:]
y_test = d_train.dropna(axis=0, how='any').iloc[:, 0] - 1
print('-' * 100)
for clf in [mlp, xgb, gbc, rfc]:
    print('当前使用模型：{}'.format(clf))
    train_and_evaluate(clf, X, X_test, y, y_test)
    print('-' * 100)
