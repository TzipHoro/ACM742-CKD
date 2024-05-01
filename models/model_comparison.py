import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.train_test import X_test, y_test
from models.ROC import ROCMetrics
from models.ckd_KNN import outcome as y_pred_knn
from models.lr_model import y_hat as y_pred_lr
from models.rf_model import y_hat as y_pred_rf

# get svm model results
loaded_svm = joblib.load('models/svm_model.pkl')
y_pred_svm = loaded_svm.predict(X_test)

# calculate roc metrics
roc = {
    'knn': ROCMetrics(y_test, y_pred_knn),
    'svm': ROCMetrics(y_test, y_pred_svm),
    'rf': ROCMetrics(y_test, y_pred_rf),
    'lr': ROCMetrics(y_test, y_pred_lr),
}

results = pd.DataFrame()

for k, v in roc.items():
    metrics = pd.Series({
        'sensitivity': v.sensitivity(),
        'specificity': v.specificity(),
        'precision': v.precision(),
        'fall_out': v.fall_out(),
        'accuracy': v.accuracy(),
        'f1_score': v.f1_score(),
    })
    results[k] = metrics

results = results.T.sort_values(['accuracy', 'f1_score'], ascending=False)

x = np.arange(4)  # the label locations
width = 0.13  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in results.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=5, fmt='%.2f')
    multiplier += 1

ax.set_title('ROC Metrics')
ax.set_xticks(x + width, results.index)
ax.legend(loc='upper right', ncols=3)
ax.set_ylim(0, 1.2)

plt.show()
