import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline

train = pd.read_csv('data/ckd_training.csv')
test = pd.read_csv('data/ckd_test.csv')

X_train = train.drop('class', axis=1)
X_test = test.drop('class', axis=1)
y_train = train['class']
y_test = test['class']

# logistic regression model
pipe = make_pipeline(preprocessing.StandardScaler(), LogisticRegression(random_state=100))
pipe.fit(X_train, y_train)
y_hat = pipe.predict(X_test)

# lr results
print(classification_report(y_test, y_hat))

# TODO: logistic regression with PCA data
