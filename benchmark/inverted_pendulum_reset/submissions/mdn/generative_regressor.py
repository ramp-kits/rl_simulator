import os

import numpy as np
from sklearn.base import BaseEstimator
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from deepnap import load_model, train_model, store_init_params

torch.manual_seed(7)

HOME = os.path.dirname(os.path.realpath(__file__))
EXP_NAME = "Best_for_saving"

load_model_from_file = False
model_file = HOME + "\\first_fold\\model_with_restart_all-data"

n_epochs = 300
LR = 1e-3
NB_LAYERS = 2
LAYER_SIZE = 200
BATCH_SIZE = 50
DECAY = 0
VAL_SIZE = 0.1

DROP_FIRST = 0
DROP_REPEATED = 1e-1
NB_GAUSSIANS = 10

MSE = nn.MSELoss()


CONST = np.sqrt(2 * np.pi)


def gauss_pdf(x, mean, sd):
    ret = torch.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * CONST)
    return ret


class CustomLoss:
    def __call__(self, y_true, y_pred):
        y_true = y_true.view(*y_true.shape, 1)
        mus = y_pred[:len(y_true)]
        sigmas = y_pred[len(y_true):len(y_true) * 2]
        w = y_pred[2 * len(y_true):]
        # one weight per component
        w_components = torch.mean(w, dim=1)
        probs = gauss_pdf(y_true, mus, sigmas)
        # multiplying over dimension each component
        multi_d_probs = torch.prod(probs, dim=1)
        # summing over components
        summed_prob = torch.sum(multi_d_probs * w_components, dim=1)
        # clamp summed_prob to avoid zeros when taking the log
        eps = torch.finfo(summed_prob.dtype).eps
        summed_prob = torch.clamp(summed_prob, min=eps)

        nll = -torch.log(summed_prob)
        nll = torch.mean(nll)
        return nll


class GenerativeRegressor(BaseEstimator):
    def __init__(self, max_dists, load_model_from_file=load_model_from_file):

        self.model_file = model_file
        if load_model_from_file:
            self.model = load_model(self.model_file)
        else:
            self.model = None
        self.max_dists = max_dists
        self.decomposition = None

    def fit(self, X_in, y_in, restart):
        if restart.ndim == 1:
            restart = np.expand_dims(restart, axis=1)
        X_in = np.hstack((X_in, restart))

        if self.model is None:
            self.model = SimpleBinnedNoBounds(NB_GAUSSIANS, X_in.shape[1],
                                              y_in.shape[1])
            dataset = torch.utils.data.TensorDataset(
                torch.Tensor(X_in), torch.Tensor(y_in))
            optimizer = optim.Adam(
                self.model.parameters(), lr=LR, weight_decay=DECAY,
                amsgrad=True)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', factor=0.1, patience=30, cooldown=20,
                min_lr=1e-7, verbose=True)
            loss = CustomLoss()
            self.model, _ = train_model(
                self.model, dataset, validation_fraction=VAL_SIZE,
                optimizer=optimizer, scheduler=scheduler,
                n_epochs=n_epochs, batch_size=BATCH_SIZE, loss_fn=loss,
                return_best_model="Loss", disable_cuda=True
                # tensorboard_path=tensorboard_path,
                )

    def predict(self, X, restart):
        X = np.array(X)

        if restart.ndim == 1:
            restart = np.expand_dims(restart, axis=1)
        X = np.hstack((X, restart))

        # we use predict sequentially in RL and there is no need to compute
        # model.eval() each time if the model is already in eval mode
        if self.model.training:
            self.model.eval()

        with torch.no_grad():
            X = torch.Tensor(X)
            n_samples = X.shape[0]
            y_pred = self.model(X)

            mus = y_pred[:n_samples].detach().numpy()
            sigmas = y_pred[n_samples:2*n_samples].detach().numpy()
            weights = y_pred[2*n_samples:].detach().numpy()

        # We put each mu next to its sigma
        params = np.empty((n_samples, mus.shape[1] * mus.shape[2] * 2))
        params[:, 0::2] = mus.reshape(n_samples, -1)
        params[:, 1::2] = sigmas.reshape(n_samples, -1)

        # gaussian has type 0
        types = ['norm'] * NB_GAUSSIANS * mus.shape[1]

        return weights.reshape(n_samples, -1), types, params


class OutputModule(nn.Module):
    def __init__(self, nb_sigmas):
        super(OutputModule, self).__init__()
        output_size_sigma = nb_sigmas
        output_size_mus = nb_sigmas
        layer_size = LAYER_SIZE
        self.mu = nn.Sequential(
            nn.Linear(layer_size, layer_size),
            torch.nn.Tanh(),
            nn.Linear(layer_size, output_size_mus)
        )

        self.sigma = nn.Sequential(
            nn.Linear(layer_size, layer_size),
            torch.nn.Tanh(),
            nn.Linear(layer_size, output_size_sigma),
        )

    def forward(self, x):
        mu = self.mu(x)
        sigma = self.sigma(x)
        sigma = torch.exp(sigma)
        return mu, sigma


class SimpleBinnedNoBounds(nn.Module):
    def __init__(self, nb_sigmas, input_size, nb_y):
        super(SimpleBinnedNoBounds, self).__init__()
        n_layers_common = NB_LAYERS
        layer_size = LAYER_SIZE
        store_init_params(self, [nb_sigmas, input_size])

        self.linear0 = nn.Linear(input_size, layer_size)
        self.act0 = nn.Tanh()
        self.drop = nn.Dropout(p=DROP_FIRST)

        self.common_block = nn.Sequential()
        for i in range(n_layers_common):
            self.common_block.add_module(
                f'layer{i + 1}-lin', nn.Linear(layer_size, layer_size))
            self.common_block.add_module(
                f'layer{i + 1}-bn', nn.BatchNorm1d(layer_size))
            self.common_block.add_module(f"layer{i + 1}-act", nn.Tanh())
            if i % 2 == 0:
                self.common_block.add_module(
                    f'layer{i + 1}-drop', nn.Dropout(p=DROP_REPEATED))

        self.detached_blocks = nn.ModuleList()
        for j in range(nb_y):
            self.detached_blocks.append(OutputModule(nb_sigmas))

        self.w = nn.Sequential(
            nn.Linear(layer_size, nb_sigmas),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.linear0(x)
        x = self.act0(x)
        raw = self.drop(x)
        x = self.common_block(raw)
        x = x + raw

        mus = []
        sigmas = []
        for block in self.detached_blocks:
            mu, sigma = block(x)
            mus.append(mu)
            sigmas.append(sigma)

        mu = torch.stack(mus, dim=1)
        sigma = torch.stack(sigmas, dim=1)

        w = self.w(x)
        w = torch.stack(mu.shape[1]*[w], dim=1)
        return torch.cat([mu, sigma, w], dim=0)
