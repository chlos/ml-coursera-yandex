#!/usr/bin/env python

import pandas
import re

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

print '\n=== 1'
# male = data[data['Sex'] == 'male'].count()['Sex']
# female = data[data['Sex'] == 'female'].count()['Sex']
counts = data['Sex'].value_counts()
print counts

print '\n=== 2'
alive = 100 * data['Survived'].value_counts() / data['Survived'].count()
print alive

print '\n=== 3'
cls = 100 * data['Pclass'].value_counts() / data['Pclass'].count()
print cls

print '\n=== 4'
mean, median = data['Age'].mean(), data['Age'].median()
print mean, median

print '\n=== 5'
corr = data['SibSp'].corr(data['Parch'])
print corr

print '\n=== 6'
female_names = list(data[data['Sex'] == 'female']['Name'])

titles = set(map(lambda x: re.match(r'^.+\s(\S+)\.', x).group(1), female_names))
print titles

first_names = []
for full_name in female_names:
    try:
        name = (re.match(r'^.+\(([a-zA-Z ].+)\)', full_name) or
                re.match(r'^.+,\s(?:Mlle|Miss|Dr|Countess|L|Mrs|Ms|Mme|lady)\.\s(.+)',
                         full_name)).group(1)
        first_name = name.split()[0]
        first_names.append(first_name)
    except AttributeError:
        print 'failed: %s' % full_name
        pass

print max(set(first_names), key=first_names.count)
first_names_pd = pandas.DataFrame({'name': first_names})
print first_names_pd['name'].value_counts()
