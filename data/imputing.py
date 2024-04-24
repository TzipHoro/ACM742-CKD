# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 00:25:44 2024

@author: bheal
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
data = pd.read_csv("C:\\Users\\bheal\\anaconda3\\envs\\myConda\\Data\\ckd_Numerical.csv",header=0)
imputer = KNNImputer(n_neighbors=5)
Final = imputer.fit_transform(data)
