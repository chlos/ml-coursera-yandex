#!/usr/bin/env python

from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
import numpy

boston = load_boston()
classes = boston['target']
features = boston['data']
features_scaled = scale(features)
kf = KFold(len(features), n_folds=5, shuffle=True, random_state=42)
res = []
for p in numpy.linspace(1, 10, 200):
    regressor = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p)
    score = cross_val_score(
        estimator=regressor,
        X=features,
        y=classes,
        scoring="mean_squared_error",
        cv=kf
    )
    res.append((p, score.mean()))
# print sorted(res, key=lambda x: x[1], reverse=True)
print sorted(res_scaled, key=lambda x: x[1], reverse=True)[0]
