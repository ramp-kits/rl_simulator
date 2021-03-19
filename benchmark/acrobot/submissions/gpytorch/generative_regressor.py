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

        self.model = None
        self.max_dists = max_dists
        self.target_dim = target_dim
        self.decomposition = 'autoregressive'

        self.scaler_y = StandardScaler()
        self.scaler_x = StandardScaler()

        print(f"Experiment number : {target_dim}")

    def fit(self, X_in, y_in):

        self.scaler_y.fit(y_in)
        self.scaler_x.fit(X_in)
        X_in = self.scaler_x.transform(X_in)
        y_in = self.scaler_y.transform(y_in)
        y_in = torch.Tensor(y_in).view(-1)

        X_in = torch.Tensor(X_in)
        if self.target_dim == 0:
            noise = 0.0001
            amplitude = 1.
            length_scales = np.array([6.72, 7.63, 0.63, 1.0, 10.36, 3.63, 0.5,
                                      7.34, 7.14])
            n_epochs = 10
        if self.target_dim == 1:
            noise = 0.0001
            amplitude = 0.7
            length_scales = np.array([0.58, 1.69, 4.95, 7.34, 16.49, 10.42,
                                      3.72, 0.6, 0.66,
                                      5.72])
            n_epochs = 10
        if self.target_dim == 2:
            noise = 0.0001
            amplitude = 1.0
            length_scales = np.array([3.71, 19.75, 2.71, 21.18, 34.79, 18.43,
                                      12.66, 4.88, 6.77, 20.6, 1.61])
            n_epochs = 10
        if self.target_dim == 3:
            noise = 0.0001
            amplitude = 1.0
            length_scales = np.array([4.19, 20.24, 3.18, 21.66, 35.3, 18.91,
                                      13.14, 5.29, 7.22, 21.08, 2.02, 1.34])
            n_epochs = 10

        GP_HYPERS['likelihood.noise_covar.noise'] = \
            torch.tensor(noise)
        GP_HYPERS['base_covar_module.outputscale'] = \
            torch.tensor(amplitude)
        GP_HYPERS['base_covar_module.base_kernel.lengthscale'] = \
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
            {'params': self.model.parameters()},
            # Includes GaussianLikelihood parameters
        ], lr=LR)
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.lk, self.model)
        for i in range(n_epochs):
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
            print(f"Iter {i + 1}/{n_epochs} - Loss: {loss.item():.3f}"
                  " - noise : "
                  f"{self.model.likelihood.noise_covar.noise.item():.3f}"
                  f" - lengthscale : {lscales}"
                  " - outputscale : "
                  f"{self.model.base_covar_module.outputscale.item():.3f}"
                  )
            optimizer.step()

    def predict(self, X):
        self.model.eval()
        X = np.array(X)

        X = self.scaler_x.transform(X)
        X_in = torch.Tensor(X)

        # Get into evaluation (predictive posterior) mode
        self.model.eval()
        self.lk.eval()

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.lk(self.model(X_in))

        mus = observed_pred.mean.view(-1, 1).detach().numpy()[:len(X_in), :]
        sigmas = observed_pred.variance.view(-1, 1).detach().numpy()
        sigmas = sigmas[:len(X_in), :]

        sigmas = np.sqrt(sigmas)

        mus = mus * self.scaler_y.scale_ + self.scaler_y.mean_
        sigmas *= self.scaler_y.scale_
        sigmas = np.abs(sigmas)

        weights = np.ones((len(X_in), 1))

        # We put each mu next to its sigma
        params = np.empty((len(X_in), N_GAUSSIANS * 2))
        params[:, 0::2] = mus
        params[:, 1::2] = sigmas

        # The last generative regressors is uniform,
        # the others are gaussians
        types = ['norm'] * N_GAUSSIANS * mus.shape[1]

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
