import numpy as np
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report as clsrpt


iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

svc = svm.LinearSVC()
svc.fit(X_train, y_train)

y_predict = svc.predict(X_test)
print(clsrpt(y_test, y_predict))
