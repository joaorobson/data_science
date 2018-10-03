import time
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
from sklearn.datasets import fetch_mldata
import seaborn as sns
from sklearn import datasets
from matplotlib import pyplot as plt

#df = pd.read_csv('a.csv') 
df = pd.read_excel('default.xls')
df = df.iloc[1:]
digits = datasets.load_digits()
X = df.drop(['Y'],axis=1)
X = df.head(10000)
x = digits.data[:500]
y = df['Y'].head(10000)
y = y.values
Y = digits.target[:500]
tsne = TSNE(n_components=3, random_state=0)

X_2d = tsne.fit_transform(X.values)
target_ids = range(len((df.Y.unique())))
print(target_ids)
print(X)

colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
k = 0
for i, c, label in zip(target_ids, colors, df.Y.unique()):	
    print(k)
    k+=1
    plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
plt.legend()
plt.show()


