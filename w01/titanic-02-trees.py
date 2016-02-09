#!/usr/bin/env python

import pandas
import numpy as np
from sklearn.tree import DecisionTreeClassifier

data = pandas.read_csv('titanic.csv',
                       index_col='PassengerId',
                       usecols=['PassengerId','Pclass', 'Fare', 'Age', 'Sex', 'Survived'])
data = data.dropna(axis=0)
data = data.replace(to_replace=['male', 'female'], value=[0, 1])
features = ['Pclass', 'Fare', 'Age', 'Sex']
X = data[features]
y = data['Survived']
# print X
# print y

clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)
importances = clf.feature_importances_
print sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
# print map(lambda x: x[0], sorted(zip(features, importances), key=lambda x: x[1], reverse=True))
