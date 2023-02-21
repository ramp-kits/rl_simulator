import numpy as np

from rampwf.utils import BaseGenerativeRegressor


class GenerativeRegressor(BaseGenerativeRegressor):
    def __init__(self, max_dists, target_dim):
        self.decomposition = 'autoregressive'

    def fit(self, X_array, y_array):
        pass

    def predict(self, X_array):
        # constant prediction with value equal to 10
        n_samples = X_array.shape[0]
        types = ['norm']
        means = np.full(shape=(n_samples, 1), fill_value=10)
        sigmas = np.zeros((n_samples, 1))

        params = np.concatenate((means, sigmas), axis=1)
        weights = np.ones((n_samples, 1))
        return weights, types, params
