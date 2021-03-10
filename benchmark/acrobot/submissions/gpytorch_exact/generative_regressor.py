import numpy as np

from sklearn.preprocessing import StandardScaler

import torch
import gpytorch

from rampwf.utils import BaseGenerativeRegressor

torch.manual_seed(7)

LR = 1e-1
N_GAUSSIANS = 1

torch.set_default_tensor_type(torch.DoubleTensor)

GP_HYPERS = {
    'likelihood.noise_covar.noise': torch.tensor(0.01),
    'covar_module.outputscale': torch.tensor(1.0),
}


class GenerativeRegressor(BaseGenerativeRegressor):
    def __init__(self, max_dists, target_dim):
        self.max_dists = max_dists
        self.target_dim = target_dim
        self.scaler_y = StandardScaler()
        self.scaler_x = StandardScaler()
        print(f"Dimension: {target_dim}")

    def fit(self, X_in, y_in):
        self.scaler_y.fit(y_in)
        self.scaler_x.fit(X_in)
        X_in = self.scaler_x.transform(X_in)
        y_in = self.scaler_y.transform(y_in)
        y_in = torch.Tensor(y_in).view(-1)
        X_in = torch.Tensor(X_in)
        if self.target_dim == 0:
            noise = 0.01
            amplitude = 0.5
            length_scales = 6 * np.ones(X_in.shape[1])
            length_scales[-1] = 0.9
            n_epochs = 50
        if self.target_dim == 1:
            noise = 0.01
            amplitude = 0.5
            length_scales = 6 * np.ones(X_in.shape[1])
            length_scales[-1] = 0.9
            n_epochs = 50
        if self.target_dim == 2:
            noise = 0.01
            amplitude = 0.5
            length_scales = 6 * np.ones(X_in.shape[1])
            length_scales[-1] = 0.9
            n_epochs = 50
        if self.target_dim == 3:
            noise = 0.01
            amplitude = 0.5
            length_scales = 6 * np.ones(X_in.shape[1])
            length_scales[-1] = 0.9
            n_epochs = 50

        GP_HYPERS['likelihood.noise_covar.noise'] =\
            torch.tensor(noise)
        GP_HYPERS['covar_module.outputscale'] =\
            torch.tensor(amplitude)
        GP_HYPERS['covar_module.base_kernel.lengthscale'] =\
            torch.tensor(length_scales)
        self.lk = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(noise))

        self.model = ExactGPModel(X_in, y_in, self.lk)
        self.model.initialize(**GP_HYPERS)

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

        for i in range(n_epochs):
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
                .numpy(),
                precision=2)
            print(
                f"Iter {i + 1}/{n_epochs} - Loss: {loss.item():.3f}"
                " - noise : "
                f"{self.model.likelihood.noise_covar.noise.item():.6f}"
                f" - lengthscale : {length_scales}"
                " - outputscale : "
                f"{self.model.covar_module.outputscale.item():.3f}"
            )

            optimizer.step()

    def predict(self, X):

        X = self.scaler_x.transform(X)
        X_in = torch.Tensor(X)
        n_samples = len(X_in)

        # Get into evaluation (predictive posterior) mode
        self.model.eval()
        self.lk.eval()

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.lk(self.model(X_in))

        mus = observed_pred.mean.view(-1, 1).detach().numpy()[:n_samples, :]
        sigmas = observed_pred.variance.view(-1, 1).detach().numpy()
        sigmas = sigmas[:n_samples, :]
        sigmas = np.sqrt(sigmas)

        mus = mus * self.scaler_y.scale_ + self.scaler_y.mean_
        sigmas *= self.scaler_y.scale_
        sigmas = np.abs(sigmas)

        weights = np.ones((n_samples, 1))

        # We put each mu next to its sigma
        params = np.empty((n_samples, N_GAUSSIANS * 2))
        params[:, 0::2] = mus
        params[:, 1::2] = sigmas
        types = ['norm'] * N_GAUSSIANS

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
