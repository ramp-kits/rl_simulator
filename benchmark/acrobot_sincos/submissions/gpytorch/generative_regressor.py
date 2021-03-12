import numpy as np

from sklearn.preprocessing import StandardScaler

import torch
import gpytorch

from rampwf.utils import BaseGenerativeRegressor

torch.manual_seed(7)

LR = 5e-2
N_GAUSSIANS = 1

torch.set_default_tensor_type(torch.DoubleTensor)

GP_HYPERS = {
    'likelihood.noise_covar.noise': torch.tensor(0.01),
    'base_covar_module.outputscale': torch.tensor(1.0),
}


class GenerativeRegressor(BaseGenerativeRegressor):
    def __init__(self, max_dists, target_dim):
        self.max_dists = max_dists
        self.model_index = target_dim
        self.decomposition = 'autoregressive'
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

        if self.model_index == 0:
            noise = 0.000004
            amplitude = 0.5
            length_scales = np.array([2.78, 2.86, 21.48, 8.26, 4.1, 15.39, 5.01])
            num_epochs = 50
        if self.model_index == 1:
            noise = 0.0001
            amplitude = 1.0
            length_scales = np.array([2.78, 2.86, 21.48, 8.26, 4.1, 15.39, 5.01, 11.13])
            num_epochs = 50
        if self.model_index == 2:
            noise = 0.0001
            amplitude = 2.5
            length_scales = np.array([1.82, 11.3, 31.83, 6.89, 13.38, 4.23, 2.75, 12.53, 1.])
            num_epochs = 50
        if self.model_index == 3:
            noise = 0.00005
            amplitude = 2.5
            length_scales = np.array([1.67, 7.4, 30.73, 6.85, 11.84, 3.33, 2.75, 11.58, 1., 1.])
            num_epochs = 50
        if self.model_index == 4:
            noise = 0.00005
            amplitude = 1.5
            length_scales = np.array([1.69, 3.2, 36.56, 11.16, 16.88, 2.97, 4.82, 11.62, 1., 1., 1.])
            num_epochs = 50
        if self.model_index == 5:
            noise = 0.00005
            amplitude = 2.5
            length_scales = np.array([2.27, 2.8, 39.75, 14.13, 14.15, 4.03, 4.68, 12.01, 1., 1., 1., 1.])
            num_epochs = 50

        GP_HYPERS['likelihood.noise_covar.noise'] =\
            torch.tensor(noise)
        GP_HYPERS['base_covar_module.outputscale'] =\
            torch.tensor(amplitude)
        GP_HYPERS['base_covar_module.base_kernel.lengthscale'] =\
            torch.tensor(length_scales)
        self.lk = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(noise)
        )
        self.model = ExactGPModel(X_in, y_in, self.lk)
        self.model.initialize(**GP_HYPERS)
        # Find optimal model hyperparameters
        self.model.train()
        self.lk.train()
        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=LR)
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.lk, self.model)
        for i in range(num_epochs):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(X_in)
            # Calc loss and backprop gradients
            loss = -mll(output, y_in).sum()
            loss.backward()
            loss.item()
            lscales = np.array2string(
                self.model.base_covar_module.base_kernel.lengthscale
                .detach().numpy(),
                precision=2)
            print(f"Iter {i + 1}/{num_epochs} - Loss: {loss.item():.3f}"
                  " - noise : "
                  f"{self.model.likelihood.noise_covar.noise.item():.3f}"
                  f" - lengthscale : {lscales}"
                  " - outputscale : "
                  f"{self.model.base_covar_module.outputscale.item():.3f}"
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
        self.base_covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]))
        inducing_indices = np.random.choice(
            range(len(train_x)), min(len(train_x), 1000), replace=False)
        self.covar_module = gpytorch.kernels.InducingPointKernel(
            self.base_covar_module,
            inducing_points=train_x[inducing_indices, :],
            likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
