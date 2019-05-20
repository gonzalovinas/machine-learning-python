import pandas as pd
from sklearn import datasets
iris = datasets.load_iris()

X_iris, y_iris = iris.data, iris.target

from sklearn.neural_network import MLPClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]


clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

print ("entrenando....")

clf.fit(X, y) 

print clf.predict([[2., 2.], [-1., -2.]])
