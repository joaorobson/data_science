import time
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
from sklearn.datasets import fetch_mldata
import seaborn as sns
from sklearn import datasets
from matplotlib import pyplot as plt

df = pd.read_excel('default.xls')
df = df.iloc[1:]

X = df.drop(['Y'],axis=1)
X = df.head(1000)

y = df['Y'].head(1000)
y = y.values

tsne = TSNE(n_components=2, random_state=0)

X_2d = tsne.fit_transform(X.values)
target_ids = range(len((df.Y.unique())))

colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
for i, c, label in zip(target_ids, colors, df.Y.unique()):	
    plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
plt.legend()
plt.show()


