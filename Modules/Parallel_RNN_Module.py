# -*- coding: utf-8 -*-
"""

Created on Tue Nov 16 09:47:28 2021

Authors:  Andreas Anton Andersen, Martin Voigt Vejling, and Morten Stig Kaaber
E-Mails: {aand17, mvejli17, mkaabe17}@student.aau.dk

In this module the main functionality supporting the training, testing, and
validation of the univariate recurrent neural network models is given, see the
report
        Forecasting Wind Power Production
            - Chapter 5: Neural Networks
            - Chapter 6: Experimental Setup
                - Section 6.2.3: Univariate RNN

The module has been developed using Python 3.9 with the
libraries numpy and pytorch.

"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, hidden_size3, dropout_hidden):
        super(LSTM, self).__init__()

        self.dropout_hidden = nn.Dropout(dropout_hidden)
        self.lstm1 = nn.LSTM(input_size, hidden_size, 1)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size2, 1)
        self.lstm3 = nn.LSTM(hidden_size2, hidden_size3, 1)
        self.fc1 = nn.Linear(hidden_size3, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.float()
        x, _ = self.lstm1(x)
        x = self.dropout_hidden(x)
        x, _ = self.lstm2(x)
        x = self.dropout_hidden(x)
        x, _ = self.lstm3(x)
        x = self.dropout_hidden(x)
        x = self.fc1(x)
        return x


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2,  hidden_size3, dropout_hidden):
        super(GRU, self).__init__()

        self.dropout_hidden = nn.Dropout(dropout_hidden)
        self.gru1 = nn.GRU(input_size, hidden_size, 1)
        self.gru2 = nn.GRU(hidden_size, hidden_size2, 1)
        self.gru3 = nn.GRU(hidden_size2, hidden_size3, 1)
        self.fc1 = nn.Linear(hidden_size3, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x, _ = self.gru1(x)
        x = self.dropout_hidden(x)
        x, _ = self.gru2(x)
        x = self.dropout_hidden(x)
        x, _ = self.gru3(x)
        x = self.dropout_hidden(x)
        x = self.fc1(x)
        return x


class PyTorchDataset(Dataset):
    def __init__(self, l, intervallength, y, z, reg, idx_power, idx_nwp):
        self.l = l
        self.intervallength = intervallength
        self.y = y
        self.z = z
        self.reg = reg
        self.idx_nwp = idx_nwp
        self.idx_power = idx_power
        self.endpoint = self.intervallength*12+1

    def __len__(self):
        return len(self.idx_power)

    def __getitem__(self, idx):
        index_p = self.idx_power[idx]
        index_n = self.idx_nwp[idx]
        predpoint = index_p + self.endpoint
        external = self.z[index_n[0],index_n[1],self.l*11:(self.l+1)*11].astype('float32')
        power = self.y[index_p:index_p+self.endpoint, self.l].astype('float32')
        if self.l < 15:
            regularisation = self.reg[predpoint, 0].astype('float32')
        else:
            regularisation = self.reg[predpoint, 1].astype('float32')
        data = np.zeros(self.endpoint+12, dtype = 'float32')
        data[0] = regularisation
        data[1:12] = external
        data[12:] = power
        target = data[-1]
        datapoint = data[:-1]
        datapoint = torch.from_numpy(datapoint)
        target = torch.from_numpy(np.array(target)).type(torch.Tensor)
        sample = (datapoint, target)
        return sample


def test(model, device, predictlength, z, y, reg, batch_size, l, intervallength, idx_Power, idx_NWP, y_max):
    """

    Evaluates the neural network in terms of MSE and NMAE.
    
    Parameters
    ----------
    model : PyTorch model class
    device : device
        Options are torch cpu or cuda.
    predictionlength : int
        Number of hours to predict ahead.
    z : ndarray, size=(n_tau_nwp, n_test_nwp, 11)
        Numerical weather prediction data.
    y : ndarray, size=(n_test, 21)
        Power data.
    reg : ndarray, size=(n_test, 2)
        Regulation data.
    batch_size : int
        The batch size used for RMSProp (stochastic gradient descent).
    l : int in [0-20]
        Integer indicating the sub-grid.
    intervallength : int
        Number of hours of history in the model
    idx_Power : dict
        Dictionary where each index translates to a valid starting index in y.
    idx_NWP : dict
        Dictionary where each index translates to a NWP used for prediction for the same index as power.
    y_max : float
        Maximum power production for each sub-grid. Used for NMAE normalisisation.
    
    Returns
    -------
    MSE : ndarray, size=(predictlength*12,)
        MSE loss for each predictionstep
    NMAE : ndarray, size=(predictlength*12,)
        NMAE loss for each predicitonstep

    """
    
    tau_max = 12*predictlength
    model.eval()
    MSE = np.zeros(tau_max, dtype = 'float32')
    NMAE = np.zeros(tau_max, dtype = 'float32')
    point = 12*intervallength
    n = 12*intervallength+12
    L = len(idx_Power)
    k = len(idx_NWP)
    if l < 15:
        reg = reg[:,0].astype('float32')
    else:
        reg = reg[:,1].astype('float32')
    with torch.no_grad():
        for i in range(k):
            a = idx_NWP[i]
            print("Batch number {} of {}".format(i+1,k))
            if i == k-1:
                batch_size = len(idx_NWP[k-1])
            for tau in range(tau_max):
                data = np.zeros((batch_size,n), dtype = 'float32')
                Power_batch = np.zeros((batch_size,point), dtype = 'float32')
                target = np.zeros(batch_size, dtype = 'float32')
                regularisation = np.zeros(batch_size, dtype = 'float32')
                for j in range(batch_size):
                    idx = idx_Power[j]
                    start = idx +tau
                    end = start + point
                    pred = end + 1
                    Power_batch[j, :] = y[start:end, l].astype('float32')
                    target[j] = y[pred, l].astype('float32')
                    regularisation[j] = reg[pred].astype('float32')
                external = z[a[:, 0]+(tau//12), a[:, 1], l*11:(l+1)*11].astype('float32')
                data[:,0] = regularisation
                data[:,1:12] = external
                data[:,12:] = Power_batch
                data, target = torch.from_numpy(data).type(torch.Tensor), torch.from_numpy(target).type(torch.Tensor)
                data, target = data.to(device), target.to(device)
                output = model(data).squeeze()
                MSE[tau] += F.mse_loss(output, target, reduction='sum').item()
                NMAE[tau] += torch.sum(torch.absolute(target - output)/y_max).item()
        MSE /= L
        NMAE /= L
    return MSE, NMAE


def validation_early_stopping(model, device, valid_loader):
    """

    Evaluates the validation set. This is done during early stopping.
    
    Parameters
    ----------
    model : PyTorch model class
    device : device
        Options are torch cpu or cuda.
    valid_loader : Dataloader
        Dataloader for the validation set.

    Returns
    -------
    valid_loss : float
        The validation loss.

    """
    
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for valid, valid_target in valid_loader:
            valid, valid_target = valid.to(device), valid_target.to(device)
            output_valid = model(valid).squeeze()
            valid_loss += F.mse_loss(output_valid, valid_target, reduction='sum').item()  # sum up batch loss
    valid_loss /= len(valid_loader.dataset)
    model.train()
    return valid_loss


def early_stopping(model, device, optimiser, scheduler, subtrain_loader,
                    valid_loader, log_interval, patience, epochs):
    """

    Determines the number of parameter updates which should be performed
    during training using early stopping.

    Parameters
    ----------
    model : PyTorch model class
    device : device
        Options are torch cpu or cuda.
    optimiser : PyTorch optimiser
    scheduler : PyTorch scheduler
    subtrain_loader : Dataloder
        Subtrain dataset.
    valid_loader : Dataloader
        Validation dataset.
    log_interval: int
        Number of parameter updates between logging the performance.
    patience : int
        Patience parameter in early stopping.
    epochs : int
        Max number of epochs to train.

    Returns
    -------
    optim_updates : int
        The optimal number of parameter updates during training
        as determined by early stopping.
    updates_pr_pretrain_epoch : int
        Number of paramer updates per epoch in the subtrain dataset.
    valid_loss_list : ndarray
        The log of validation losses. This is saved to a .csv file.
    training_loss_list : ndarray
        The log of training losses. This is saved to a .csv file.
    min_valid_loss : float
        The minimum validation loss observed.

    """
    
    updates_counter = 0
    min_valid_loss = 0
    no_increase_counter = 0
    optim_updates = 0
    updates_pr_pretrain_epoch = len(subtrain_loader)
    valid_loss_list = []
    training_loss_list = []
    for epoch in range(1, epochs + 1):
        model.train()
        interval_loss = 0
        for batch_idx, (data, target) in enumerate(subtrain_loader):
            data, target = data.to(device), target.to(device)

            target = target.squeeze()
            optimiser.zero_grad()
            output = model(data).squeeze()
            loss = F.mse_loss(output, target, reduction='mean')
            loss.backward()
            optimiser.step()

            interval_loss += loss.item()
            if batch_idx % log_interval == 0 and batch_idx != 0:
                valid_loss = validation_early_stopping(model, device, valid_loader)
                valid_loss_list.append(valid_loss)
                training_loss_list.append(interval_loss/log_interval)
                
                if min_valid_loss == 0:
                    min_valid_loss = valid_loss
                elif valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    optim_updates = updates_counter
                    no_increase_counter = 0
                else:
                    no_increase_counter += 1

                print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}\tMin. Val. Loss: {:.6f}\tVal. Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(subtrain_loader.dataset),
                    interval_loss/log_interval, min_valid_loss, valid_loss))
                interval_loss = 0

                if no_increase_counter == patience:
                    return optim_updates, updates_pr_pretrain_epoch, valid_loss_list, training_loss_list, min_valid_loss

            updates_counter += 1
        scheduler.step()
        print('')
        if no_increase_counter == patience:
            break
    return optim_updates, updates_pr_pretrain_epoch, valid_loss_list, training_loss_list, min_valid_loss


def early_stopping_retrain(model, device, train_loader, optimiser, epoch,
                           optim_updates, updates_counter, scheduler,
                           updates_pr_pretrain_epoch, log_interval):
    """

    Re-trains the neural network after early stopping pre-training.
    The learning rate is decayed after each updates_pr_pretrain_epoch 
    parameter updates and training is done for optim_updates
    parameter updates.

    Parameters
    ----------
    model : PyTorch model class
    device : device
        Options are torch cpu or cuda.
    train_loader : Dataloader
        Training dataset.
    optimiser : PyTorch optimiser
    epoch : int
        The current training epoch.
    optim_updates : int
        The optimal number of parameter updates during training
        as determined by early stopping.
    updates_counter : int
        Counter for the number of parameter updates.
    scheduler : PyTorch scheduler
    updates_pr_pretrain_epoch : int
        Number of paramer updates per epoch in the subtrain dataset.
    log_interval : int 
        Number of parameter updates between logging the performance.

    Returns
    -------
    updates_counter : int
        Counter for the number of parameter updates.

    """
    
    model.train()
    interval_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        target = target.squeeze()
        optimiser.zero_grad()
        output = model(data).squeeze()
        loss = F.mse_loss(output, target)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        interval_loss += loss.item()
        if batch_idx % log_interval == 0 and batch_idx != 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), interval_loss/log_interval))
            interval_loss = 0

        if updates_counter == optim_updates:
            return updates_counter
        updates_counter += 1
        if updates_counter % updates_pr_pretrain_epoch == 0:
            scheduler.step()
    return updates_counter
