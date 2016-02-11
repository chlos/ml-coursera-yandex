#!/usr/bin/env python

import pandas
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data_train = pandas.read_csv('perceptron-train.csv', header=None)
y_train = data_train[0]
X_train = data_train.iloc[0:, 1:]
data_test = pandas.read_csv('perceptron-test.csv', header=None)
y_test = data_test[0]
X_test = data_test.iloc[0:, 1:]

clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
res = accuracy_score(y_test, pred)
print 'res: %s' % res

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
clf = Perceptron(random_state=241)
clf.fit(X_train_scaled, y_train)
pred = clf.predict(X_test_scaled)
res_scaled = accuracy_score(y_test, pred)
print 'res_scaled: %s' % res_scaled

print 'diff: %s' % (res_scaled - res)
