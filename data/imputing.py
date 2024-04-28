# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 00:25:44 2024

@author: bheal
"""

import pandas as pd
from sklearn.impute import KNNImputer

data = pd.read_csv("ckd_numerical.csv", header=0)
imputer = KNNImputer(n_neighbors=5)
Final = imputer.fit_transform(data)
