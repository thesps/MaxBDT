from sklearn.datasets import make_moons
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from DFEBDT import DFEBDT
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

# Convert features to fixed point format and run on DFE
Z_DFE = DFEBDT(n * n, (np.array(list(chain.from_iterable(izip(xx.ravel(), yy.ravel())))) * 2**24).astype('int').tolist())
#Z_DFE = DFEBDT((xx.ravel() * 2**24).astype('int').tolist(), (yy.ravel() * 2**24).astype('int').tolist())
Z_DFE = (np.array(Z_DFE) * 2**-16).reshape(xx.shape)
# Run on CPU
Z_CPU = bdt.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z_CPU = Z_CPU.reshape(xx.shape)

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

# Plot the DFE decision contour
ax = plt.subplot(1, 2, 1)
ax.contourf(xx, yy, Z_DFE, cmap=cm, alpha=.8)
ax.set_title('DFE decision contour')
#Plot the training points
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
# and testing points
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())

# Repeat for CPU
ax = plt.subplot(1, 2, 2)
ax.contourf(xx, yy, Z_CPU, cmap=cm, alpha=.8)
ax.set_title('CPU decision contour')
# Plot the training points
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
# and testing points
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())

plt.tight_layout()
plt.show()

