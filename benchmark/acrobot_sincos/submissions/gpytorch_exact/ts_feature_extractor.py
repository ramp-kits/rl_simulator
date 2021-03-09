FEATURE_NAMES = [
    'theta_dot_2', 'theta_dot_1', 'torque',
    'cos_theta_1', 'sin_theta_1', 'cos_theta_2', 'sin_theta_2']


class FeatureExtractor:
    def __init__(self, restart_name, n_burn_in=0, n_lookahead=1):
        """
        Parameters
        ----------
        restart_name : str
            The name of the 0/1 column indicating restarts in the time series.
        """
        self.restart_name = restart_name
        self.n_burn_in = n_burn_in
        self.n_lookahead = n_lookahead

    def transform(self, X_df):
        """Transform time series into list of states.
        We use the observables at time t as the state

        Be careful not to use any information from the future (X_ds[t + 1:])
        when constructing X_df[t].
        Parameters
        ----------
        X_df_raw : xarray.Dataset
            The raw time series.
        Return
        ------
        X_df : pandas Dataframe

        """
        return X_df[FEATURE_NAMES]