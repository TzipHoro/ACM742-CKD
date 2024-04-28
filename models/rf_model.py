import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

train = pd.read_csv('data/ckd_training.csv')
test = pd.read_csv('data/ckd_test.csv')

X_train = train.drop('class', axis=1)
X_test = test.drop('class', axis=1)
y_train = train['class']
y_test = test['class']

# random forest model
mod1 = RandomForestClassifier(random_state=100)
mod1.fit(X_train, y_train)
y_hat = mod1.predict(X_test)

# plot one of the trees
plt.figure(figsize=(20, 10))
tree.plot_tree(mod1.estimators_[0],
               filled=True,
               rounded=True,
               feature_names=X_train.columns)
plt.show()

# rf results
print(classification_report(y_test, y_hat))

# TODO: random forest with PCA data
