# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 18:00:52 2024

@author: bheal
"""

from sklearn.neighbors import KNeighborsClassifier

from data.train_test import X_train, y_train, X_test

# train the model
neighbor = KNeighborsClassifier(n_neighbors=5)
neighbor.fit(X_train, y_train)

# Predict
outcome = neighbor.predict(X_test)
