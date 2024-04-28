# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 18:00:52 2024

@author: bheal
"""

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# set-up data
Train = pd.read_csv("ckd_training.csv", header=0)
Test = pd.read_csv("ckd_test.csv", header=0)
Train = Train.to_numpy()
Test = Test.to_numpy()
X_train = Train[:, 0:24]
y_train = Train[:, 24:25]
X_test = Test[:, 0:24]
y_test = Test[:, 24:25]

# train the model
neighbor = KNeighborsClassifier(n_neighbors=5)
neighbor.fit(X_train, y_train)

# Predict
outcome = neighbor.predict(X_test)
