from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.nn import functional as F

from rampwf.utils import BaseGenerativeRegressor

from mbrltools.pytorch_utils import train

n_epochs = 100
LR = 1e-3
N_LAYERS = 2
LAYER_SIZE = 60
BATCH_SIZE = 20
REPRESENTATION_SIZE = 5
DECAY = 0
DROP_REPEATED = 0
KLD_W = 1
VAL_SIZE = 0.15


class CustomLoss:
    def __call__(self, y, model_out):
        data_sample, mu, log_var = model_out
        recon_loss = F.mse_loss(data_sample, y)
        kl_loss = torch.mean(-0.5 * torch.sum(
            1 + log_var - mu ** 2 - log_var.exp(), dim=1
        ), dim=0)

        loss = recon_loss + KLD_W * kl_loss
        return loss


class GenerativeRegressor(BaseGenerativeRegressor):
    def __init__(self, max_dists, target_dim):
        self.max_dists = max_dists
        self.scaler_y = StandardScaler()
        self.decomposition = None

    def fit(self, X_in, y_in):

        # standardize features
        self.std = StandardScaler().fit(X_in)
        X_in = self.std.transform(X_in)
        self.scaler_y.fit(y_in)
        y_in = self.scaler_y.transform(y_in)
        self.model = VAE(y_in.shape[1], X_in.shape[1])
        dataset = torch.utils.data.TensorDataset(
            torch.Tensor(X_in), torch.Tensor(y_in))
        optimizer = optim.Adam(
            self.model.parameters(), lr=LR, weight_decay=DECAY,
            amsgrad=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=0.1, patience=30, cooldown=20,
            min_lr=1e-7, verbose=True)
        loss = CustomLoss()
        self.model, _ = train(
            self.model, dataset, validation_fraction=VAL_SIZE,
            optimizer=optimizer, scheduler=scheduler,
            n_epochs=n_epochs, batch_size=BATCH_SIZE, loss_fn=loss,
            return_best_model="Loss", disable_cuda=True, drop_last=True,
            is_vae=True
        )

    def sample(self, X, rng=None):
        # we use predict sequentially in RL and there is no need to compute
        # model.eval() each time if the model is already in eval mode
        if self.model.training:
            self.model.eval()

        X = self.std.transform(X)

        with torch.no_grad():
            X = torch.Tensor(X)
            y_pred = self.model.sample(X)

        y_pred = y_pred * self.scaler_y.scale_ + self.scaler_y.mean_

        return y_pred.detach().numpy()


class VAE(nn.Module):
    def __init__(self, input_size, cond_size):
        super(VAE, self).__init__()
        self.representation_size = REPRESENTATION_SIZE
        self.input_size = input_size
        self.cond_size = cond_size

        self.en_in = nn.Sequential(
            nn.Linear(input_size + cond_size, LAYER_SIZE),
            torch.nn.Tanh(),
        )

        self.en = nn.Sequential()
        for i in range(N_LAYERS):
            self.en.add_module(
                f'layer{i + 1}-lin', nn.Linear(LAYER_SIZE, LAYER_SIZE))
            # self.common_block.add_module(
            #     f'layer{i + 1}-bn', nn.BatchNorm1d(layer_size))
            self.en.add_module(f"layer{i + 1}-act", nn.Tanh())
            self.en.add_module(
                f'layer{i + 1}-drop', nn.Dropout(p=DROP_REPEATED))

        self.en_mu = nn.Linear(LAYER_SIZE, self.representation_size)
        self.en_std = nn.Linear(LAYER_SIZE, self.representation_size)

        self.de_in = nn.Sequential(
            nn.Linear(self.representation_size + cond_size, LAYER_SIZE),
            torch.nn.Tanh()
        )

        self.de = nn.Sequential()
        for i in range(N_LAYERS):
            self.de.add_module(
                f'layer{i + 1}-lin', nn.Linear(LAYER_SIZE, LAYER_SIZE))
            self.de.add_module(f"layer{i + 1}-act", nn.Tanh())
            self.de.add_module(
                f'layer{i + 1}-drop', nn.Dropout(p=DROP_REPEATED))

        self.de_out = nn.Linear(LAYER_SIZE, input_size)

    def encode(self, y, x):
        """Encode a batch of samples.

        Return posterior parameters for each point.
        """
        data = torch.cat([y, x], axis=1)
        h = self.en_in(data)
        h = self.en(h)
        return self.en_mu(h), self.en_std(h)

    def decode(self, z, x):
        """Decode a batch of latent variables"""
        data = torch.cat([z, x], axis=1)
        data = self.de_in(data)
        data = self.de(data)
        return self.de_out(data)

    def reparam(self, mu, logvar):
        """Reparametrisation trick to sample z values.

        This is stochastic during training, and returns the mode during
        evaluation.
        """
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, y, x):
        """Takes a batch of samples, encodes them, and then decodes them again.
        """
        mu, logvar = self.encode(y, x)
        z = self.reparam(mu, logvar)
        return self.decode(z, x), mu, logvar

    def sample(self, x):
        """Encode a batch of data points, x, into their z representations."""
        z = Variable(torch.randn(len(x), self.representation_size))
        return self.decode(z, x)
