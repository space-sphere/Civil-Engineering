import numpy as np
import pandas as pd

class data_normalize:
    def __init__(self, data: pd.DataFrame, tg: str):
        self.data = data.copy()
        self.tg = tg

    def fillna_by_avg(self):
        y = self.data[self.tg].unique().tolist()
        df = self.data.copy()
        for i in y:
            aver = np.mean(df[df[self.tg]==i], axis=0)
            df[df[self.tg]==i] = df[df[self.tg]==i].fillna(aver)
        return df

    def fillna_by_random(self):
        y = self.data[self.tg].unique().tolist()
        df = self.data
        row_id = np.sum(self.data.isna(), axis=0) == 0
        x = self.data[row_id[row_id == 0].index.tolist()]
        for i in y:
            for col in x.columns:
                # s = df[df[self.tg]==i][col].unique().tolist()  # 找出所有符合条件的值，从中选择一个填入空值
                df1 = df[df[self.tg]==i].copy()
                if np.sum(df1[col].isna() == 0) > 0:
                    for j in df1[df1[col].isna()].index:
                        df.loc[j, col] = np.random.choice(df1[df1[col].isna() == 0][col])
                else:
                    k = 1
                    df2 = df[df[self.tg]==i-k].copy()
                    while np.sum(df2[col].isna() == 0) == 0 & k < i:
                        print(k)
                        k += 1
                        df2 = df[df[self.tg]==i-k].copy()
                    for j in df1[df1[col].isna()].index:
                        df.loc[j, col] = np.random.choice(df2[df2[col].isna() == False][col]) * i / (i - k)
        return df

    def iso(self, df):
        m, n = df.shape
        df1 =  df.copy()
        for i in range(n):
            df1.iloc[:, i] = pd.qcut(df.iloc[:, i].rank(method='first'), 5, labels=False)
        return df1

    def fillna_by_random_fix(self):  # 使用相同WLC的数据进行补充
        y = self.data[self.tg].unique().tolist()
        df = self.data.copy()
        row_id = np.sum(self.data.isna(), axis=0) == 0
        x = self.data[row_id[row_id == 0].index.tolist()]
        for i in y:
            for col in x.columns:
                # s = df[df[self.tg]==i][col].unique().tolist()  # 找出所有符合条件的值，从中选择一个填入空值
                df1 = df[df[self.tg] == i].copy()
                if np.sum(df1[col].isna() == 0) > 0:
                    for j in df1[df1[col].isna()].index:
                        if np.sum(df1[df1['WLC'] == df.loc[j, 'WLC']][col].isna() == 0) > 0:
                            df3 = df1[df1['WLC'] == df.loc[j, 'WLC']]
                            df.loc[j, col] = np.random.choice(df3[df3[col].isna() == 0][col])
                        else:
                            df.loc[j, col] = np.random.choice(df1[df1[col].isna() == 0][col])
                else:
                    k = 1
                    df2 = df[df[self.tg] == i - k].copy()
                    while np.sum(df2[col].isna() == 0) == 0 & k < i:
                        print(k)
                        k += 1
                        df2 = df[df[self.tg] == i - k].copy()
                    for j in df1[df1[col].isna()].index:
                        df.loc[j, col] = np.random.choice(df2[df2[col].isna() == False][col]) * i / (i - k)
        return df

    def fillna_by_randoml(self):
        y = self.data[self.tg].unique().tolist()
        df = self.data.copy()
        row_id = np.sum(self.data.isna(), axis=0) == 0
        x = self.data[row_id[row_id == 0].index.tolist()]
        for i in y:
            for col in x.columns:
                # s = df[df[self.tg]==i][col].unique().tolist()  # 找出所有符合条件的值，从中选择一个填入空值
                df1 = df[df[self.tg] == i].copy()
                if np.sum(df1[col].isna() == 0) > 0:
                    for j in df1[df1[col].isna()].index:
                        if np.sum(df1[df1['WLC'] == df.loc[j, 'WLC']][col].isna() == 0) > 0:
                            df3 = df1[df1['WLC'] == df.loc[j, 'WLC']]
                            df.loc[j, col] = np.random.choice(df3[df3[col].isna() == 0][col])
                        else:
                            df.loc[j, col] = np.random.choice(df1[df1[col].isna() == 0][col])
                else:
                    k = 1
                    df2 = df[df[self.tg] == i - k].copy()
                    while np.sum(df2[col].isna() == 0) == 0 & k < i:
                        print(k)
                        k += 1
                        df2 = df[df[self.tg] == i - k].copy()
                    for j in df1[df1[col].isna()].index:
                        df.loc[j, col] = np.random.choice(df2[df2[col].isna() == False][col]) * i / (i - k)
        return df


if __name__ == '__main__':
    df = pd.read_excel(r'D:\program\pycharm\model\Data\相关性最新.xlsx', sheet_name='Sheet1')
    data = data_normalize(df, tg='WD')
    print(data.fillna_by_random())
    print(df)
