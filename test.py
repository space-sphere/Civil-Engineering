import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
df = pd.read_csv(r"D:\program\pycharm\model\Data\out.csv")

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes[0, 1].set_title('WLC')
sns.distplot(np.array(df['WLC']), ax=axes[0, 0])
# sns.distplot(np.array(df['WLC']))
# axs[0, 1] = sns.distplot(df['WLA'])
# axs[0, 2] = sns.distplot(df['WIA'])
# axs[0, 3] = sns.distplot(df['TD'])
# axs[0, 4] = sns.distplot(df['TI'])

# axs[0, 1] = sns.distplot([1, 2, 3])

plt.show()