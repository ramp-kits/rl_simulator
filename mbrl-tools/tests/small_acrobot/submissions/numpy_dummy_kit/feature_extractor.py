class FeatureExtractor:
    def __init__(self):
        pass

    def transform(self, X):
        X[:, [0, 3]] = X[:, [3, 0]]
        return X
