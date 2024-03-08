import numpy as np
import numpy.random as npr
import datetime

"""
该算法需要算力非常大，基本无法在较短时间内得出结果，有改进的方式，可能有些复杂，不知道需不需要？
"""

class Individual:
    _n = 0
    eval = 0.0
    chromsome = None

    def __init__(self, n):
        self._n = n
        v = []
        v.append(np.random.choice([100 + 10 * i for i in range(25)]))  # n_estimators 0
        v.append(npr.randint(1, 11))  # max_depth 1
        v.append(npr.randint(1, 11))  # min_samples 2
        v.append(npr.uniform(1e-3, 5e-1))  #  learning_rate 3
        v.append(npr.randint(1, 11))  # max_features 4
        v.append(npr.uniform(5e-1, 1))  # subsample 5
        self.chromsome = v

    def crossover(self, another):
        startPos = npr.randint(self._n)  # 交叉的起始位置
        jeneLength = npr.randint(self._n) + 1  # 交叉的长度
        son1 = Individual(self._n)
        son2 = Individual(self._n)
        son1.chromsome = self.chromsome.copy()
        son2.chromsome = another.chromsome.copy()
        endpos = startPos + jeneLength
        son1.chromsome[startPos:endpos] = another.chromsome[startPos:endpos]
        son2.chromsome[startPos:endpos] = self.chromsome[startPos:endpos]
        return son1, son2

    def mutation(self, learnRate):
        son = Individual(self._n)
        son.chromsome = self.chromsome.copy()
        mutationPos = npr.randint(self._n)
        if mutationPos in [1, 2, 4]:
            son.chromsome[mutationPos] = npr.randint(1, 11)
        # 产生一个-0.5-0.5之间的随机小数
        elif mutationPos == 0:
            son.chromsome[mutationPos] += learnRate * (npr.randint(200) - 100)
        elif mutationPos == 3:
            son.chromsome[mutationPos] *= 1 - learnRate * (npr.randint(2) * 2 - 1)
        else:
            temp = npr.random() - 0.5
            son.chromsome[mutationPos] += learnRate / 2 * temp
            if son.chromsome[mutationPos] > 1:
                son.chromsome[mutationPos] = 1
        return son


class NGA:
    population = []
    dimension = 1
    bestPos = worstPos = 0
    mutationProb = 10
    crossoverProb = 90
    maxIterTime = 1000
    evalFunc = None
    arfa = 1.0
    popu = 2

    def __init__(self, popuNum,evalFunc, dimension=6, crossoverProb=10, mutationProb=90, maxIterTime=1000):
        for i in range(popuNum):
            date1 = datetime.datetime.now()
            oneInd = Individual(dimension)
            oneInd.eval = evalFunc(oneInd.chromsome)
            self.population.append(oneInd)
            date2 = datetime.datetime.now()
            print('当前第{}个单体，用时为{}秒'.format(i, (date2 - date1).seconds))

        self.crossoverProb = crossoverProb
        self.mutationProb = mutationProb
        self.maxIterTime = maxIterTime
        self.evalFunc = evalFunc
        self.popu = popuNum
        self.dimension = dimension

    # 找最好的个体位置
    def findBestWorst(self):
        self.population.sort(key=lambda o: o.eval)
        self.bestPos = 0
        self.worstPos = self.popu - 1

    # 交叉操作
    def crossover(self):
        fatherPos = npr.randint(0, self.popu)
        motherPos = npr.randint(0, self.popu)
        while motherPos == fatherPos:
            motherPos = npr.randint(0, self.popu)
        father = self.population[fatherPos]
        mother = self.population[motherPos]
        son1, son2 = father.crossover(mother)

        son1.eval = self.evalFunc(son1.chromsome)  # ;// 评估第一个子代
        son2.eval = self.evalFunc(son2.chromsome)
        self.findBestWorst()

        if son1.eval < self.population[self.worstPos].eval:
            self.population[self.worstPos] = son1
        self.findBestWorst()
        if son2.eval < self.population[self.worstPos].eval:
            self.population[self.worstPos] = son2

    def mutation(self):
        father = self.population[npr.randint(self.popu)]
        son = father.mutation(self.arfa)
        son.eval = self.evalFunc(son.chromsome)
        self.findBestWorst()
        if son.eval < self.population[self.worstPos].eval:
            self.population[self.worstPos] = son

    def solve(self):
        shrinkTimes = self.maxIterTime / 10
        # //将总迭代代数分成10份
        oneFold = shrinkTimes  # ;//每份中包含的次数
        i = 0
        while i < self.maxIterTime:
            print(i, "---", self.maxIterTime)
            if i == shrinkTimes:
                self.arfa = self.arfa / 2.0
                # 经过一份代数的迭代后，将收敛参数arfa缩小为原来的1/2，以控制mutation
                shrinkTimes += oneFold  # ;//下一份到达的位置
            for j in range(self.crossoverProb):
                self.crossover()
            for j in range(self.mutationProb):
                self.mutation()
            print("solution:", self.population[self.bestPos].chromsome)
            print("func value:", self.population[self.bestPos].eval)
            i = i + 1

    def getAnswer(self):
        self.findBestWorst()
        return self.population[0].chromsome


if __name__ == '__main__':
    import pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    import warnings
    warnings.filterwarnings('ignore')
    df = pd.read_csv(r"D:\program\pycharm\model\Data\out.csv")
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    def nga_objective(params):
        clf = GradientBoostingClassifier(
            n_estimators=int(params[0]),
            max_depth=int(params[1]),
            min_samples_split=int(params[2]),
            learning_rate=params[3],
            max_features=int(params[4]),
            loss='deviance',
            subsample=params[5],
            random_state=0
        )
        scores = np.mean(cross_val_score(clf, X, y, scoring='accuracy', cv=5))
        return 1 - scores
    print('start_solving')
    nga = NGA(30, nga_objective)
    print('individual generated')
    nga.solve()
    ans = nga.getAnswer()
    print(ans)
    print(nga_objective(ans))
