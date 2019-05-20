import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt

colors = ['red', 'greenyellow', 'blue']

iris = datasets.load_iris()

X_iris, y_iris = iris.data, iris.target

X, y = X_iris[:, :2], y_iris


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

print X_train.shape, y_train.shape

scaler = preprocessing.StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

for i in xrange(len(colors)):
    xs = X_train[:, 0][y_train == i]
    ys = X_train[:, 1][y_train == i]
    plt.scatter(xs, ys, c=colors[i])

plt.legend(iris.target_names)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()