# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 17:04:17 2024

@author: bheal
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

#import data
data = pd.read_csv("C:\\Users\\bheal\\anaconda3\\envs\\myConda\\Data\\ckd_Numerical.csv",header=0)

#KNN imputer
imputer = KNNImputer(n_neighbors=5)
imputed = imputer.fit_transform(data).round(1) #round so that we fit binary categories

#Set-up test and training data
X = imputed[:, 0:24]
y = imputed[:, 24:25]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#Bootstrap by resampling the training data
combo_train = np.hstack((X_train, y_train))
Training = resample(combo_train, n_samples=1000)