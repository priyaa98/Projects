# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 16:21:32 2019

@author: Priyamvadha
"""

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

##Creating dummy Random Forest model to find best parameters
rf_dummy=RandomForestClassifier()

##Finding best parameters using RandomizedSearchCV
##Parameters to be chosen based on the dataset and the type of machine learning model to be used
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 500, num = 10)]
max_features = ["auto", "sqrt"]
max_depth = [int(x) for x in np.linspace(100, 500, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True]
criterion=["gini","entropy"]
oob_score=[True,False]
warm_start=[True,False]
class_weight=[None,"balanced"]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'criterion': criterion,
               'oob_score': oob_score,
               'warm_start': warm_start,
               'class_weight': class_weight
               }

##Testing for best parameters
##Can be performed for any machine learning algorithm
rf_random = RandomizedSearchCV(estimator = rf_dummy, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, 
                               random_state=3, n_jobs = -1)
##Fitting for the training set
##To be performed on desired dataset
rf_random.fit(training_set_features,training_set_labels)
##Best parameters
print(rf_random.best_params_)