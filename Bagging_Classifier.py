# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:55:15 2026

@author: haris
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn import model_selection
from sklearn import ensemble


os.chdir("H:/VijaNaar/titanic/Submissions")

titanic_train = pd.read_csv("train.csv")
titanic_train.info()

titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
titanic_train1.info()

X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'], axis=1, inplace=False)
y_train = titanic_train['Survived']

dt_estimator = tree.DecisionTreeClassifier()

bag_tree_estimator1 = ensemble.BaggingClassifier(estimator = dt_estimator, n_estimators=5)
scores = model_selection.cross_val_score(bag_tree_estimator1, X_train, y_train, cv = 3)

bag_tree_estimator1.fit(X_train, y_train)

print(scores)
print(scores.mean())

bag_tree_estimator2 = ensemble.BaggingClassifier(estimator = dt_estimator, n_estimators = 6, random_state = 5)
bag_grid = {'Criterion': ['entropy', 'gini']}

bag_grid_estimator = model_selection.GridSearchCV(bag_tree_estimator2, bag_grid, n_jobs=6)
bag_tree_estimator2.fit(X_train, y_train)

'''
n_tree = 0
for est in bag_tree_estimator2.estimators_:
    dot_data = io.StringIO()
    tree.export_graphviz(est, out_file = dot_data, feature_names = X_train.columns)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("bagtree"+str(n_tree)+".pdf")
    n_tree = n_tree+1
'''
n_tree = 0
for est in bag_tree_estimator2.estimators_:
    plt.figure(figsize=(60,30))
    tree.plot_tree(est, feature_names = X_train.columns, filled=True, rounded=True, fontsize=6)
    plt.title(f"Decission Tree {n_tree}")
    plt.savefig(f"bagtree{n_tree}.pdf")
    plt.close()
    n_tree = n_tree + 1

    
os.getcwd()