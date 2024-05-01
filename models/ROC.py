"""https://github.com/TzipHoro/NeuralNetworks-FinalProject/blob/master/ROC.py"""
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_curve, auc, RocCurveDisplay, confusion_matrix, f1_score


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
plt.style.use('seaborn-v0_8')


class ROCMetrics:
    def __init__(self, y_true: pd.Series, y_pred: pd.Series):
        self.y_true = y_true
        self.y_pred = y_pred
        self.confusion_matrix = self.conf_matrix()

    def conf_matrix(self) -> np.array:
        return confusion_matrix(self.y_true, self.y_pred)

    def sensitivity(self) -> float:
        tp = self.confusion_matrix[1][1]
        fn = self.confusion_matrix[1][0]

        with np.errstate(divide='ignore', invalid='ignore'):
            sens = np.true_divide(tp, tp + fn)
            if sens == np.inf:
                sens = 0

        return sens

    def specificity(self) -> float:
        tn = self.confusion_matrix[0][0]
        fp = self.confusion_matrix[0][1]

        with np.errstate(divide='ignore', invalid='ignore'):
            spec = np.true_divide(tn, tn + fp)
            if spec == np.inf:
                spec = 0

        return spec

    def precision(self) -> float:
        tp = self.confusion_matrix[1][1]
        fp = self.confusion_matrix[0][1]

        with np.errstate(divide='ignore', invalid='ignore'):
            prec = np.true_divide(tp, tp + fp)
            if prec == np.inf:
                prec = 0

        return prec

    def fall_out(self) -> float:
        tn = self.confusion_matrix[0][0]
        fp = self.confusion_matrix[0][1]

        with np.errstate(divide='ignore', invalid='ignore'):
            fo = np.true_divide(fp, tn + fp)
            if fo == np.inf:
                fo = 0

        return fo

    def accuracy(self) -> float:
        return accuracy_score(self.y_true, self.y_pred)

    def f1_score(self) -> float:
        return f1_score(self.y_true, self.y_pred)
