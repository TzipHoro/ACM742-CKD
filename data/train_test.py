import pandas as pd

train = pd.read_csv('data/ckd_training.csv')
test = pd.read_csv('data/ckd_test.csv')

X_train = train.drop('class', axis=1)
X_test = test.drop('class', axis=1)
y_train = train['class']
y_test = test['class']
