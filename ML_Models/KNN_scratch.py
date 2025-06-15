import numpy as np 
from collections import Counter

def euc_dist(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self,X,y):
        self.X_train = X
        self.y_train = y
        pass

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        distances = [euc_dist(x,x_train) for x_train in self.X_train]

        k_indices = np.argsort(distances)[:self.k]  #return indices that would sort an array if we use that order in original array
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        most_common_label = Counter(k_nearest_labels).most_common(1)
        return most_common_label[0][0]

    