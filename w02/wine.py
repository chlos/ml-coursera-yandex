#!/usr/bin/env python

import pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale

data = pandas.read_csv('wine.data', header=None)
classes = data[0]
# print classes
features = data.iloc[0:, 1:]
# print features


kf = KFold(len(features), n_folds=5, shuffle=True, random_state=42)

res_not_scaled = []
res_scaled = []
for k in xrange(1, 50 + 1):
    classifier = KNeighborsClassifier(n_neighbors=k)

    cv_results_not_scaled = cross_val_score(
        estimator=classifier,
        X=features,
        # y=y.transpose()[0],
        y=classes,
        scoring="accuracy",
        cv=kf
    )
    res_not_scaled.append((k, cv_results_not_scaled.mean()))

    features_scaled = scale(features)
    cv_results_scaled = cross_val_score(
        estimator=classifier,
        X=features_scaled,
        # y=y.transpose()[0],
        y=classes,
        scoring="accuracy",
        cv=kf
    )
    res_scaled.append((k, cv_results_scaled.mean()))
print '===== not scaled'
print sorted(res_not_scaled, key=lambda x: x[1], reverse=True)[0]
print '===== scaled'
print sorted(res_scaled, key=lambda x: x[1], reverse=True)[0]
