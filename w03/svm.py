#!/usr/bin/env python

import pandas
from sklearn.svm import SVC

data = pandas.read_csv('svm-data.csv', header=None)
y = data[0]
X = data.iloc[0:, 1:]

classifier = SVC(C=100000, kernel='linear', random_state=241)
classifier.fit(X, y)
print map(lambda x: x + 1, classifier.support_)
