FEATURE_NAMES = [
    'theta_dot_2', 'theta_dot_1', 'torque',
    'cos_theta_1', 'sin_theta_1', 'cos_theta_2', 'sin_theta_2']


class FeatureExtractor:
    def __init__(self):
        pass

    def fit(self, X_array, y_array):
        pass

    def transform(self, X_array):
        """
        Parameters
        ----------
        X_array : numpy array
            Inputs.
        Return
        ------
        X_tf : numpy array

        """
        X_tf = X_array[:, :len(FEATURE_NAMES)]
        # to match acrobot_sincos without numpy
        X_tf[:, [0, 3]] = X_tf[:, [3, 0]]
        return X_tf
