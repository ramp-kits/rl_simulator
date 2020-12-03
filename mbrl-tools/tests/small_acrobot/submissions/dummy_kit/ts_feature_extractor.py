class FeatureExtractor:
    def __init__(self, restart_name, n_burn_in, n_lookahead):
        self.restart_name = restart_name
        self.n_burn_in = n_burn_in
        self.n_lookahead = n_lookahead

    def transform(self, X_df_raw):
        X_df = X_df_raw.copy()
        return X_df[['theta_dot_2', 'theta_2', 'theta_dot_1', 'theta_1']]
