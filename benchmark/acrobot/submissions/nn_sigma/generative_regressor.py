import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from rampwf.utils import BaseGenerativeRegressor

from mbrltools.pytorch_utils import train

n_epochs = 100
batch_size = 200
validation_fraction = 0.05


class GenerativeRegressor(BaseGenerativeRegressor):
    def __init__(self, max_dists, target_dim):
        """
        Parameters
        ----------
        max_dists : int
            The maximum number of distributions (kernels) in the mixture.
        target_dim : int
            The index of the target column to be predicted.
        """
        self.max_dists = max_dists
        self.sigma = None
        self.decomposition = 'autoregressive'

    def fit(self, X, y):
        """
        Pytorch simple regressor to estimate the mu of a gaussian,
        we estimate the sigma over the training set.
        """
        self.model = PytorchReg(X.shape[1])
        dataset = torch.utils.data.TensorDataset(
            torch.Tensor(X), torch.Tensor(y))
        optimizer = optim.Adam(self.model.parameters(), lr=4e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=0.1, patience=30, verbose=True)

        train(
            self.model, dataset, validation_fraction=validation_fraction,
            return_best_model=True, optimizer=optimizer,
            scheduler=scheduler, n_epochs=n_epochs, batch_size=batch_size,
            loss_fn=nn.MSELoss(), disable_cuda=True)

        # we run our model over the whole training data to get an estimate
        # of sigma from the residual variance
        self.model.eval()
        with torch.no_grad():
            X = torch.Tensor(X)
            y_guess = self.model(X).detach().numpy()

        error = (y - y_guess).ravel()
        self.sigma = np.sqrt((1 / (X.shape[0] - 1)) * np.sum(error ** 2))

    def predict(self, X):
        """Construct a conditional mixture distribution.
        Return
        ------
        weights : np.array of float
            discrete probabilities of each component of the mixture
        types : np.array of int
            integer codes referring to component types
            see rampwf.utils.distributions_dict
        params : np.array of float tuples
            parameters for each component in the mixture
        """
        # we use predict sequentially in RL and there is no need to compute
        # model.eval() each time if the model is already in eval mode
        if self.model.training:
            self.model.eval()

        with torch.no_grad():
            X = torch.Tensor(X)
            y_pred = self.model(X)

        n_samples = X.shape[0]
        sigmas = np.full(shape=(n_samples, 1), fill_value=self.sigma)

        params = np.concatenate((y_pred, sigmas), axis=1)
        types = ['norm']  # Gaussian
        weights = np.ones((n_samples, 1))

        return weights, types, params


class PytorchReg(nn.Module):
    def __init__(self, input_size):
        super(PytorchReg, self).__init__()
        n_layers_common = 2
        layer_size = 200

        self.linear0 = nn.Linear(input_size, layer_size)
        self.act0 = nn.LeakyReLU()
        self.common_block = nn.Sequential()
        for i in range(n_layers_common):
            self.common_block.add_module(
                f"layer{i + 1}-lin", nn.Linear(layer_size, layer_size))
            self.common_block.add_module(
                f"layer{i + 1}-act", nn.LeakyReLU())
        self.linear_end = nn.Linear(layer_size, layer_size // 2)
        self.act_end = nn.LeakyReLU()
        self.linear_out = nn.Linear(layer_size // 2, 1)

    def forward(self, x):
        x = self.linear0(x)
        x = self.act0(x)
        x = self.common_block(x)
        x = self.linear_end(x)
        x = self.act_end(x)
        x = self.linear_out(x)
        return x
