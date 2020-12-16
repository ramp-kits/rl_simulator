import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.utils.data
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from deepnap.utils import _set_device
import json
import torch.nn.init as init
from deepnap import *

import gpytorch
import os

HOME = os.path.dirname(os.path.realpath(__file__))

DECAY = 0
LR = 5e-2
BATCH = 1000
device = "cpu"
VAL_SIZE = 0.15

PATIENCE = 15
NB_GAUSSIANS = 1

torch.manual_seed(7)
torch.set_default_tensor_type(torch.DoubleTensor)

GP_HYPERS = {
    'likelihood.noise_covar.noise': torch.tensor(0.01),
    'covar_module.outputscale': torch.tensor(1.0),
}


class GenerativeRegressor(BaseEstimator):
    def __init__(self, max_dists, model_index):
        self.model = None
        self.max_dists = max_dists
        self.model_index = model_index
        self.scaler_y = StandardScaler()
        self.scaler_x = StandardScaler()
        print(f"Dimension : {model_index}")

    def fit(self, X_in, y_in):

        self.scaler_y.fit(y_in)
        self.scaler_x.fit(X_in)
        X_in = self.scaler_x.transform(X_in)
        y_in = self.scaler_y.transform(y_in)
        y_in = torch.Tensor(y_in).view(-1)
        X_in = torch.Tensor(X_in)
        if self.model is None:
            if self.model_index == 0:
                noise = 0.000004
                amplitude = 0.5
                length_scales = np.array([2.78, 2.86, 21.48, 8.26, 4.1, 15.39, 5.01, 11.13])
                num_epochs = 1
            if self.model_index == 1:
                noise = 0.0001
                amplitude = 1.0
                length_scales = np.array([2.78, 2.86, 21.48, 8.26, 4.1, 15.39, 5.01, 11.13])
                num_epochs = 1
            if self.model_index == 2:
                noise = 0.0001
                amplitude = 2.5
                length_scales = np.array([1.82, 11.3, 31.83, 6.89, 13.38, 4.23, 2.75, 12.53])
                num_epochs = 1
            if self.model_index == 3:
                noise = 0.00005
                amplitude = 2.5
                length_scales = np.array([1.67, 7.4, 30.73, 6.85, 11.84, 3.33, 2.75, 11.58])
                num_epochs = 1
            if self.model_index == 4:
                noise = 0.00005
                amplitude = 1.5
                length_scales = np.array([1.69, 3.2, 36.56, 11.16, 16.88, 2.97, 4.82, 11.62])
                num_epochs = 1
            if self.model_index == 5:
                noise = 0.00005
                amplitude = 2.5
                length_scales = np.array([2.27, 2.8, 39.75, 14.13, 14.15, 4.03, 4.68, 12.01])
                num_epochs = 1

            GP_HYPERS['likelihood.noise_covar.noise'] =\
                torch.tensor(noise)
            GP_HYPERS['covar_module.outputscale'] =\
                torch.tensor(amplitude)
            GP_HYPERS['covar_module.base_kernel.lengthscale'] =\
                torch.tensor(length_scales)
            self.lk = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(noise)
            )

            self.model = ExactGPModel(X_in, y_in, self.lk)

            self.model.initialize(**GP_HYPERS)
            # clip = 0.1
            # torch.nn.utils.clip_grad_norm(self.model.parameters(), clip)

            # Find optimal model hyperparameters
            self.model.train()
            self.lk.train()

            # Use the adam optimizer
            optimizer = torch.optim.Adam([
                # Includes GaussianLikelihood parameters
                {'params': self.model.parameters()},
            ], lr=LR)

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.lk, self.model)

            for i in range(num_epochs):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = self.model(X_in)
                # Calc loss and backprop gradients
                loss = -mll(output, y_in).sum()
                loss.backward()
                loss.item()

                length_scales = np.array2string(
                    self.model.covar_module.base_kernel.lengthscale.detach()
                    .numpy(), precision = 2)
                print(
                    f"Iter {i + 1}/{num_epochs} - Loss: {loss.item():.3f}"
                    f" - noise : {self.model.likelihood.noise_covar.noise.item():.6f}"
                    f" - lengthscale : {length_scales}"
                    f" - outputscale : {self.model.covar_module.outputscale.item():.3f}"
                )

                optimizer.step()

    def predict(self, X):
        self.model.eval()

        X = self.scaler_x.transform(X)
        X_in = torch.Tensor(X)
        batch = BATCH
        num_data = X.shape[0]
        num_batches = int(num_data / batch) + 1

        # Get into evaluation (predictive posterior) mode
        self.model.eval()
        self.lk.eval()

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.lk(self.model(X_in))

        mus = observed_pred.mean.view(-1, 1).detach().numpy(
            )[:len(X_in), :]
        sigmas = observed_pred.variance.view(-1, 1).detach().numpy(
            )[:len(X_in), :]
        sigmas = np.sqrt(sigmas)

        mus = mus * self.scaler_y.scale_ + self.scaler_y.mean_
        sigmas *= self.scaler_y.scale_
        sigmas = np.abs(sigmas)

        weights = np.ones((len(X_in), 1))

        # We put each mu next to its sigma
        params = np.empty((len(X_in), NB_GAUSSIANS * 2))
        params[:, 0::2] = mus
        params[:, 1::2] = sigmas

        # The last generative regressors is uniform,
        # the others are gaussians
        types = np.zeros(NB_GAUSSIANS)
        types = np.array([types] * len(X_in))

        return weights, types, params


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
