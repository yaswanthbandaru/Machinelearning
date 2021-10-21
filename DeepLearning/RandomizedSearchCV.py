# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 20:19:18 2021

Use scikit-learn's RandomizedSearchCV

@author: yash
"""

#Load Libraries
from scipy.stats import uniform
from sklearn import linear_model, datasets
from sklearn.model_selection import RandomizedSearchCV

# Load Data
iris = datasets.load_iris()
features = iris.data
target = iris.target 

# create logistic regression
logistic = linear_model.LogisticRegression()

# create range of candidate regurlaization hyperparameter values
penalty = ['l1', 'l2']

# create distribution of candidate regularization hyperparameter values
C = uniform(loc=0, scale=4)

# create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)

# create randomized search
randomizedsearch = RandomizedSearchCV(logistic, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=0, n_jobs=-1)

# Fit randomized search
best_model = randomizedsearch.fit(features, target)
