#!/usr/bin/env python

import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV

newsgroups = datasets.fetch_20newsgroups(
    subset='all',
    categories=['alt.atheism', 'sci.space']
)
X = newsgroups.data
y = newsgroups.target

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(X)

# grid = {'C': np.power(10.0, np.arange(-5, 6))}
# cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241)
# clf = SVC(kernel='linear', random_state=241)
# gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
# gs.fit(tfidf, y)
# for a in gs.grid_scores_:
    # print a.mean_validation_score
    # print a.parameters

c = 1.0
clf = SVC(C=c, kernel='linear', random_state=241)
clf.fit(tfidf, y)

indices = clf.coef_.indices
data = map(abs, clf.coef_.data)
top10_indices = map(lambda x: x[1],
                    sorted(zip(data, indices), reverse=True)[:10]
)
tfidf_words = tfidf_vectorizer.get_feature_names()
top10_words = sorted([tfidf_words[i] for i in top10_indices])
print top10_words
