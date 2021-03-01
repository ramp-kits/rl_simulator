import copy

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data


def _set_device(disable_cuda=False):
    """Set device to CPU or GPU.

    Parameters
    ----------
    disable_cuda : bool (default=False)
        Whether to use CPU instead of GPU.

    Returns
    -------
    device : torch.device object
        Device to use (CPU or GPU).
    """
    # XXX we might also want to use CUDA_VISIBLE_DEVICES if it is set
    if not disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    return device


def train(model, dataset_train, dataset_valid=None,
          validation_fraction=None,
          n_epochs=10, batch_size=128, loss_fn=nn.MSELoss(),
          optimizer=None, scheduler=None, return_best_model=False,
          disable_cuda=False,
          batch_size_predict=None, drop_last=False):
    """Training model using the provided dataset and given loss function.

    Parameters
    ----------
    model : pytorch nn.Module
        Model to be trained.
    dataset_train : Tensor dataset.
        Training data set.
    dataset_valid : Tensor dataset.
        If not None, data set used to compute a validation loss. This data set
        is not used to train the model.
    validation_fraction : float in (0, 1).
        If not None, fraction of samples from dataset to put aside to be
        use as a validation set. If dataset_valid is not None then
        dataset_valid overrides validation_fraction.
    n_epochs : int
        Number of epochs
    batch_size : int
        Batch size.
    loss_fn : function
        Pytorch loss function.
    optimizer : object
        Pytorch optimizer
    scheduler : object
        Pytorch scheduler.
    return_best_model : bool
        Whether to return the best model on the validation loss. More exactly,
        if set to True, the model trained at the epoch that leads to the best
        performance on the validation dataset is returned. In this case the
        best validation loss is also returned.
    disable_cuda : bool
        Whether to use CPU instead of GPU.
    batch_size_predict : int
        Batch size to use for the computation of the validation loss
        in case of a very large valid dataset. If None, no batch size is used.
    drop_last : bool
        Whether to drop the last batch in the dataloader if incomplete.

    Returns
    -------
    model : pytorch nn.Module
        Trained model. If return_best_model is set to True the best validation
        loss is also returned.
    """
    # use GPU by default if cuda is available, otherwise use CPU
    device = _set_device(disable_cuda=disable_cuda)
    model = model.to(device)

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # dataset_valid has priority over validation_fraction. if no dataset_valid
    # but validation fraction then build dataset_train and dataset_valid
    if dataset_valid is None and validation_fraction is not None:
        # split dataset into a training and validation set
        if validation_fraction <= 0 or validation_fraction >= 1:
            raise ValueError('validation_fraction should be in (0, 1).')

        n_samples = len(dataset_train)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        ind_split = int(np.floor(validation_fraction * n_samples))
        train_indices, val_indices = indices[ind_split:], indices[:ind_split]
        dataset_valid = data.TensorDataset(*dataset_train[val_indices])
        dataset_train = data.TensorDataset(*dataset_train[train_indices])

    if dataset_valid is not None:
        X_valid = dataset_valid.tensors[0]
        y_valid = dataset_valid.tensors[1]

        if return_best_model:
            best_val_loss = np.inf

    dataset_train = data.DataLoader(dataset_train, batch_size=batch_size,
                                    shuffle=True, drop_last=drop_last)
    n_train = len(dataset_train.dataset)

    val_scheduler = isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau)

    for epoch in range(n_epochs):

        model.train()

        if scheduler is not None and not val_scheduler:
            scheduler.step()

        train_loss = 0

        # training
        for (idx, (x, y)) in enumerate(dataset_train):
            x, y = x.to(device), y.to(device)
            x, y = Variable(x), Variable(y)
            model.zero_grad()

            out = model(x)
            loss = loss_fn(y, out)

            # update training loss
            train_loss += len(x) * loss.item()

            # backward and optimization
            loss.backward()
            optimizer.step()

        train_loss /= n_train

        if dataset_valid is None:
            print('[{}/{}] Training loss: {:.4f}'
                  .format(epoch, n_epochs-1, train_loss))
        else:
            print('[{}/{}] Training loss: {:.4f}'
                  .format(epoch, n_epochs-1, train_loss), end='\t')

        # loss on validation set
        if dataset_valid is not None:
            y_valid_pred = predict(model, X_valid,
                                   batch_size=batch_size_predict,
                                   disable_cuda=disable_cuda, verbose=0)
            val_loss = loss_fn(y_valid, y_valid_pred).item()
            print('Validation loss: {:.4f}'.format(val_loss))

            if val_scheduler:
                scheduler.step(val_loss)

            if return_best_model:
                if val_loss < best_val_loss:
                    best_model = copy.deepcopy(model)  # XXX I don't like this
                    best_val_loss = val_loss

    # return best model and best val loss if we want it
    if (dataset_valid is not None) and return_best_model:

        if n_epochs == 0:  # we return the passed model
            best_model = model
            y_valid_pred = predict(best_model, X_valid,
                                   batch_size=batch_size_predict,
                                   disable_cuda=disable_cuda, verbose=0)
            best_val_loss = loss_fn(y_valid, y_valid_pred).item()

        return best_model, best_val_loss

    return model


def predict(model, dataset, batch_size=None, disable_cuda=False, verbose=0):
    """Predict outputs of dataset using trained model"""

    if batch_size is None:
        batch_size = len(dataset)

    model.eval()
    device = _set_device(disable_cuda=disable_cuda)

    dataset = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = []
    with torch.no_grad():
        for i, x in enumerate(dataset):
            x = x.to(device)
            predictions.append(model.forward(x).cpu())

            if verbose > 1 and i % 100 == 0:
                print('[{}/{}]'.format(i, len(dataset)))

    return torch.cat(predictions, dim=0)
