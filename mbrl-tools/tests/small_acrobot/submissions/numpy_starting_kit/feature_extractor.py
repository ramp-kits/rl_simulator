import numpy as np

FEATURE_NAMES = ['theta_1', 'theta_dot_1', 'theta_2', 'theta_dot_2', 'torque']


class FeatureExtractor:
    def __init__(self):
        pass

    def transform(self, X):
        """Transform.

        Parameters
        ----------
        X : numpy array
            Inputs.
        Return
        ------
        X : numpy array
        """
        cos_sin = []
        for i in range(2):
            theta_i = X[:, i]
            cos_sin.append(np.cos(theta_i).reshape(-1, 1))
            cos_sin.append(np.sin(theta_i).reshape(-1, 1))

        cos_sin_array = np.concatenate(cos_sin, axis=1)
        X = np.concatenate([X[:, :len(FEATURE_NAMES)], cos_sin_array], axis=1)
        X[:, [0, 3]] = X[:, [3, 0]]  # to match acrobot_ts_generative_regression kit

        return X
