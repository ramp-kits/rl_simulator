import numpy as np
from numpy.testing import assert_almost_equal

import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim

from mbrltools.pytorch_utils import train, predict, _set_device

torch.manual_seed(0)


class MLP(nn.Module):
    """Multi-Layer Perceptron for the sake of testing."""

    def __init__(self, input_size, output_size, layers,
                 activation=nn.LeakyReLU()):
        super(MLP, self).__init__()

        model = nn.Sequential()
        model.add_module('initial-lin', nn.Linear(input_size, layers[0]))
        model.add_module('initial-act', activation)

        for i in range(len(layers) - 1):
            model.add_module('layer{}-lin'.format(i + 1),
                             nn.Linear(layers[i], layers[i + 1]))
            model.add_module('layer{}-act'.format(i + 1), activation)

        model.add_module('final-lin', nn.Linear(layers[-1], output_size))

        self.model = model

    def forward(self, x):
        return self.model(x)


def test_batch_predict():
    # Batch predict is equal to non batch predict

    # create a simple model (a MLP with 2 inner layers)
    device = _set_device()
    input_size, output_size, layers = 2, 2, [5, 5]
    model = MLP(input_size, output_size, layers)
    model = model.to(device)

    # create a random dataset. take a number of samples that is not a multiple
    # of the batch_size.
    n_samples = 26
    x = torch.randn(n_samples, input_size)
    y = torch.randn(n_samples, output_size)
    dataset = data.TensorDataset(x, y)
    model = train(model, dataset, n_epochs=1, batch_size=10)

    with torch.no_grad():
        predictions_without_batch = model(x.to(device)).cpu()
        predictions_with_batch = predict(model, x, batch_size=10)
        predictions_with_predict_no_batch = predict(model, x, batch_size=None)
        assert_almost_equal(
            predictions_with_batch.numpy(), predictions_without_batch.numpy())
        assert_almost_equal(
            predictions_with_batch.numpy(),
            predictions_with_predict_no_batch.numpy())


def test_best_model():
    # check the best model
    loss_fn = torch.nn.MSELoss()

    # create a simple model (a MLP with 2 inner layers)
    device = _set_device()
    input_size, output_size, layers = 2, 2, [50, 50]
    model = MLP(input_size, output_size, layers)
    model = model.to(device)

    # create a random dataset
    n_samples = 100
    x = torch.randn(n_samples, input_size)
    y = 2 * x + 1
    x, y = x.to(device), y.to(device)
    dataset = data.TensorDataset(x, y)

    # create training and validation sets
    validation_fraction = 0.5
    n_samples = len(dataset)
    ind_split = int(np.floor(validation_fraction * n_samples))
    dataset_train = data.TensorDataset(*dataset[ind_split:])
    dataset_valid = data.TensorDataset(*dataset[:ind_split])

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    model, best_val_loss = train(
        model, dataset_train, optimizer=optimizer, dataset_valid=dataset_valid,
        n_epochs=10, batch_size=20, return_best_model=True, loss_fn=loss_fn)

    # check val_loss
    X_valid = dataset_valid.tensors[0]
    y_valid = dataset_valid.tensors[1]
    y_valid_pred = predict(model, X_valid)
    val_loss = loss_fn(y_valid, y_valid_pred).item()
    assert_almost_equal(val_loss, best_val_loss)
