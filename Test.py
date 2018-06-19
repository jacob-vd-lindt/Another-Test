import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class AdalineH(object):

    def __init__(self, alpha=0.01, numIter=15, randomState=1):
        self.alpha = alpha
        self.numIter = numIter
        self.randomState = randomState

    def fit(self, X, y):
        ones = np.ones((np.size(y), 1))
        X = np.hstack((ones, X))        #add row of ones to the front of X
        rangen = np.random.RandomState(self.randomState)
        self.w_ = rangen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.cost_ = []

        for i in range(self.numIter):
            netInput = self.netInput(X)
            errors = (y - netInput)

            self.w_ = self.w_ + self.alpha * X.T.dot(errors)
            cost = (errors**2).sum()/2.0
            print(cost)

            self.cost_.append(cost)
        return self

    def netInput(self, X):
        return np.dot(X, self.w_)

    def predict(self, X):
        ones = np.ones((np.size(X, 0), 1))
        X = np.hstack((ones, X))
        return np.where(self.netInput(X) >= 0.0, 1, -1)

#-----------------------------------------------------------------------------------------------------------------------


#get X and y traininig sets.

df = pd.read_csv('https://archive.ics.uci.edu/ml/''machine-learning-databases/iris/iris.data', header=None)
y = df.iloc[0:100, 4]
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

Adda1 = AdalineH(alpha=0.00001, numIter=1055).fit(X, y)

#-------------------------------------------------------------------------------------------------------------------------
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and colormap
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    #plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y= X[y == cl, 1], alpha=0.8, c=colors[idx], marker=markers[idx], label=cl, edgecolor='black')
#---------------------------------------------------------------------------------------------------------------------------------

plot_decision_regions(X, y, classifier=Adda1)
plt.title('Adaline - Gradient Descent')
plt.xlabel('Sepal Length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

plt.plot(range(1, len(Adda1.cost_) + 1), Adda1.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('cost, sum-squared error')
plt.show()