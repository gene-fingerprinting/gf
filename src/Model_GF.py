import const
import numpy as np
import pandas as pd

from tqdm import tqdm
from os.path import join, exists
from utility import load_obj, save_obj
from joblib import Parallel, delayed
from collections import Counter
from dtaidistance.dtw import distance

saved_model = join(const.MODEL_DIR, 'saved_model_GF.pkl')

class GF:
    def __init__(self, exemplar, K, window, mode, quantile=None, n_jobs=None):
        self.exemplar = exemplar
        self.K = K
        self.window = window
        self.mode = mode
        self.n_jobs = n_jobs
        self.quantile = quantile
    
    def train(self, X_train, y_train):
        if not exists(saved_model):
            data = pd.DataFrame({'X': X_train, 'y': y_train})
            X_exemplar = []
            y_exemplar = []
            for y_i in tqdm(data['y'].unique(), desc='Electing'):
                samples = data[data['y'] == y_i].copy()
                samples['sum'] = samples['X'].apply(lambda X: sum([distance(X_i, X, window=self.window * min(len(X_i), len(X))) for X_i in samples['X']]))
                exemplars = samples.sort_values('sum')[:self.exemplar]
                X_exemplar += exemplars['X'].tolist()
                y_exemplar += exemplars['y'].tolist()
            self.X = X_exemplar
            self.y = y_exemplar
            save_obj({'X': self.X, 'y': self.y}, saved_model)
        else:
            model = load_obj(saved_model)
            self.X = model['X']
            self.y = model['y']

        if self.mode == const.MODE_OW:
            self.threshold(X_train, y_train)

    def threshold(self, X_train, y_train):
        dists = self.min_distances(X_train)
        min_dist = [dist.min() for dist in dists]
        min_removed = []
        y = np.array(y_train)
        ys = np.unique(y_train)
        for y_i in ys:
            min_removed += [min([dist.min() for dist in y_dists[ys != y_i]]) for y_dists in dists[y == y_i]]
        
        fitter = pd.DataFrame({'y': y_train, 'min_dist': min_dist, 'min_removed': min_removed})
        thresholds = {}
        for y_i in ys:
            y_dists = fitter[fitter['y'] == y_i]
            lower_dist = y_dists['min_dist'].quantile(q=self.quantile)
            upper_dist = y_dists['min_removed'].quantile(q=1-self.quantile)
            thresholds[y_i] = (lower_dist + upper_dist) / 2
        self.thresholds = thresholds
    
    def test(self, X_test, y_test):
        if self.mode == const.MODE_CW:
            self.accuracy(X_test, y_test)
        else:
            self.metrics(X_test, y_test)
    
    def accuracy(self, X_test, y_test):
        y_predict = self.predict(X_test)
        correct = sum(np.array(y_predict) == np.array(y_test))
        accuracy = correct / len(y_test)
        print('Accuracy: {:.2f} %'.format(100 * accuracy))
    
    def metrics(self, X_test, y_test):
        dists = self.min_distances(X_test)
        y_predict = [dist.argmin() if dist.min() < self.thresholds[dist.argmin()] else const.MON_NUM for dist in dists]

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
    
    def predict(self, X_test):
        y_predict = Parallel(n_jobs=self.n_jobs)(delayed(self._predict)(X_i) for X_i in tqdm(X_test, desc='Predict'))
        return y_predict

    def _predict(self, X):
        dists = [distance(X_i, X, window=self.window * min(len(X_i), len(X))) for X_i in self.X]
        nearest = np.argsort(dists)
        top_k = [self.y[i] for i in nearest[:self.K]]
        votes = Counter(top_k)
        return votes.most_common(1)[0][0]

    def min_distances(self, X_test):
        dists = Parallel(n_jobs=self.n_jobs)(delayed(self._min_distances)(X_i) for X_i in tqdm(X_test, desc='Distance'))
        return np.array(dists)

    def _min_distances(self, X):
        dists = [distance(X_i, X, window=self.window * min(len(X_i), len(X))) for X_i in self.X]
        min_dists = [min(dists[y*self.exemplar:y*self.exemplar+self.exemplar]) for y in np.unique(self.y)]
        return np.array(min_dists)
