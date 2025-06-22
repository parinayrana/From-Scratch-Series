import numpy as np


class NaiveBayes:

    def fit(self, X,y):
        n_sample, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes,n_features), dtype=np.float64)
        self._var = np.zeros((n_classes,n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for c in self._classes:
            X_c = X[y==c]
            self._mean[c,:] 

    def predict(self,X):
        pass