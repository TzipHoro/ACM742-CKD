import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from sklearn.pipeline import make_pipeline

from data.train_test import X_train, y_train, X_test, y_test

# logistic regression model
pipe = make_pipeline(preprocessing.StandardScaler(), LogisticRegression(random_state=100))
pipe.fit(X_train, y_train)
y_hat = pipe.predict(X_test)

pipe = make_pipeline(preprocessing.StandardScaler(), PCA(n_components=10, random_state=100), LogisticRegression(random_state=100))
pipe.fit(X_train, y_train)
y_hat_pca = pipe.predict(X_test)

if __name__ == '__main__':
    # lr results
    print(classification_report(y_test, y_hat))

    fpr, tpr, thresholds = roc_curve(y_test, y_hat)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False')
    plt.ylabel('True')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()

    cm = confusion_matrix(y_test, y_hat)

    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # logistic regression with PCA data
    # lr results
    print(classification_report(y_test, y_hat_pca))

    fpr, tpr, thresholds = roc_curve(y_test, y_hat_pca)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False')
    plt.ylabel('True')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()

    cm = confusion_matrix(y_test, y_hat_pca)

    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
