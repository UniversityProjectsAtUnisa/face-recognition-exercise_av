from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklPCA
import numpy as np
from .plot_module import *
from itertools import count


class PCA:
    def __init__(self, target_variance=0.999):
        self.target_variance = target_variance
        self.pca = None
        self.scaler = StandardScaler()

    def fit(self, X_train):
        X_train_std = self.scaler.fit_transform(X_train)
        variance = 0
        for K in count(1):
            pca = sklPCA(n_components=K)
            pca.fit(X_train_std)
            variance = np.sum(pca.explained_variance_ratio_)
            if variance >= self.target_variance:
                self.pca = pca
                return self.pca

    def __call__(self, X):
        if self.pca is None:
            return None
        X_train_std = self.scaler.transform(X)
        return self.pca.transform(X_train_std)
