import const
import numpy as np
import pandas as pd

from os.path import join, exists
from utility import load_obj, save_obj
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

saved_model = join(const.MODEL_DIR, 'saved_model_CUMUL.pkl')

def optimizer(X, y):
    param_grid = [{ 
     'C': [2**11, 2**13, 2**15, 2**17],
     'gamma': [2**-3, 2**-1, 2**1, 2**3]
    }]
    clf = GridSearchCV(estimator=SVC(kernel='rbf'), param_grid=param_grid, \
                       scoring='accuracy', cv=5, verbose=0, n_jobs=-1)
    scaler = preprocessing.MinMaxScaler((-1, 1))
    X = scaler.fit_transform(X)
    clf.fit(X, y)
    C = clf.best_params_['C']
    gamma = clf.best_params_['gamma']
    return C, gamma


class CUMUL:
    def __init__(self, C, gamma, mode, quantile=None):
        self.model = SVC(C=C, kernel='rbf', gamma=gamma, probability=True)
        self.scaler = preprocessing.MinMaxScaler((-1, 1))
        self.mode = mode
        self.quantile = quantile

    def train(self, X_train, y_train):
        X_train = self.scaler.fit_transform(X_train)
        if not exists(saved_model):
            self.model.fit(X_train, y_train)
            save_obj(self.model, saved_model)
        else:
            self.model = load_obj(saved_model)

        if self.mode == const.MODE_OW:
            self.threshold(X_train, y_train)

    def threshold(self, X_train, y_train):
        probas = self.model.predict_proba(X_train)
        max_proba = [proba.max() for proba in probas]
        max_removed = []
        y = np.array(y_train)
        ys = np.unique(y_train)
        for y_i in ys:
            max_removed += [max([proba.max() for proba in y_probas[ys != y_i]]) for y_probas in probas[y == y_i]]
        
        fitter = pd.DataFrame({'y': y_train, 'max_proba': max_proba, 'max_removed': max_removed})
        thresholds = {}
        for y_i in ys:
            y_probas = fitter[fitter['y'] == y_i]
            upper_proba = y_probas['max_proba'].quantile(q=self.quantile)
            lower_proba = y_probas['max_removed'].quantile(q=1-self.quantile)
            thresholds[y_i] = (upper_proba + lower_proba) / 2
        self.thresholds = thresholds

    def test(self, X_test, y_test):
        X_test = self.scaler.transform(X_test)
        if self.mode == const.MODE_CW:
            self.accuracy(X_test, y_test)
        else:
            self.metrics(X_test, y_test)
            
    def accuracy(self, X_test, y_test):
        y_predict = self.model.predict(X_test)
        correct = sum(np.array(y_predict) == np.array(y_test))
        accuracy = correct / len(y_test)
        print('Accuracy: {:.2f} %'.format(100 * accuracy))

    def metrics(self, X_test, y_test):
        probas = self.model.predict_proba(X_test)
        y_predict = [proba.argmax() if proba.max() > self.thresholds[proba.argmax()] else const.MON_NUM for proba in probas]

        result = pd.DataFrame({'truth': y_test, 'predict': y_predict})
        TP = sum((result['truth'] < const.MON_NUM) & (result['predict'] < const.MON_NUM))
        FN = sum((result['truth'] < const.MON_NUM) & (result['predict'] == const.MON_NUM))
        FP = sum((result['truth'] >= const.MON_NUM) & (result['predict'] < const.MON_NUM))
        TN = sum((result['truth'] >= const.MON_NUM) & (result['predict'] == const.MON_NUM))

        TPR = TP / (TP + FN)
        FNR = FN / (TP + FN)
        FPR = FP / (FP + TN)
        Precision = TP / (TP + FP)
        print('TPR: {:.2f} %'.format(100 * TPR), end='\t')
        print('FNR: {:.2f} %'.format(100 * FNR), end='\t')
        print('FPR: {:.2f} %'.format(100 * FPR), end='\t')
        print('Precision: {:.2f} %'.format(100 * Precision))
