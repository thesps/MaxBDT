import timeit

setup='''
from sklearn.datasets import make_moons
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import bitstring as bs
import numpy as np
from DFEBDTGalava import DFEBDTGalava
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from itertools import chain, izip

X, y = make_moons(noise=0.3, random_state=0)

bdt = joblib.load('bdt.pkl') 
X_train, X_test, y_train, y_test =\
  train_test_split(X, y, test_size=.4, random_state=42)


# Make a mesh of features
n = 128
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
dx = (x_max - x_min) / n
dy = (y_max - y_min) / n
xx, yy = np.meshgrid(np.arange(x_min, x_max, dx),
                     np.arange(y_min, y_max, dy))
dfedata = (np.array(list(chain.from_iterable(izip(xx.ravel(), yy.ravel())))) * 2**24).astype('int').tolist()
'''
# Convert features to fixed point format and run on DFE
print timeit.timeit("DFEBDTGalava(n * n, dfedata)", setup=setup, number=1)
# Run on CPU
print timeit.timeit('bdt.decision_function(np.c_[xx.ravel(), yy.ravel()])', setup=setup, number=1)
#Z_CPU = Z_CPU.reshape(xx.shape)


