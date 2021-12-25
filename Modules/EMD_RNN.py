# -*- coding: utf-8 -*-
"""

Created on Tue Nov 30 15:07:57 2021

Authors:  Andreas Anton Andersen, Martin Voigt Vejling, and Morten Stig Kaaber
E-Mails: {aand17, mvejli17, mkaabe17}@student.aau.dk

In this module functionality supporting the use of recurrent neural networks
for the EMD-LSTM-ARMA model is given. This includes handling not including
numerical weather prediction data in the model as well as a specific formatting
of testing to be compatible with the ARMA models. See the report
        Forecasting Wind Power Production
            - Chapter 6: Experimental Setup
                - Section 6.2.6: EMD-LSTM-ARMA

The module has been developed using Python 3.9 with the
libraries numpy and pytorch.

"""

import numpy as np
import torch
from torch.utils.data import Dataset


class PyTorchDataset(Dataset):
    def __init__(self, intervallength, IMF, reg_data, area, idx_power, reg = True):
        self.area = area
        self.intervallength = intervallength
        self.IMF = IMF
        self.reg_data = reg_data
        self.idx_power = idx_power
        self.endpoint = self.intervallength*12+1
        self.reg = reg

    def __len__(self):
        return len(self.idx_power)

    def __getitem__(self, idx):
        if self.reg == True:
            index_p = self.idx_power[idx]
            predpoint = index_p + self.endpoint
            power = self.IMF[index_p:index_p+self.endpoint].astype('float32')
            if self.area<15:
                regularisation = self.reg_data[predpoint, 0].astype('float32')
            else:
                regularisation = self.reg_data[predpoint, 1].astype('float32')
            data = np.zeros(self.endpoint+1, dtype = 'float32')
            data[0] = regularisation
            data[1:] = power
            target = data[-1]
            datapoint = data[:-1]
            datapoint = torch.from_numpy(datapoint)
            target = torch.from_numpy(np.array(target)).type(torch.Tensor)
            sample = (datapoint, target)
        else:    
            index_p = self.idx_power[idx]
            predpoint = index_p + self.endpoint
            power = self.IMF[index_p:index_p+self.endpoint].astype('float32')
            target = power[-1]
            datapoint = power[:-1]
            datapoint = torch.from_numpy(datapoint)
            target = torch.from_numpy(np.array(target)).type(torch.Tensor)
            sample = (datapoint, target)
        return sample


def test(model, device, predictlength, IMF, reg_test, batch_size, area, intervallength, idx_Power, reg = True):
    """
    Evaluates the neural network in terms of MSE and NMAE.
    
    Parameters
    ----------
    model : PyTorch model class
    device : device
    predictionlength : int
        Number of hours we predict ahead
    IMF : ndarray
        an IMF of the Power data
    reg_test : ndarray
        regularisation data
    batch_size : int
        Batch size
    area : int in [0-20]
        integer telling us which wind area we are in only used if reg = True
    intervallength : int
        Number of days of history in the model
    idx_Power : dict
        Dictionary where each index translates to a valid starting index in y
    idx_NWP : dict
        Dictionary where each index translates to a NWP used for prediction for the same index as power
    y_max : float
        maximum power production for each wind area. Used for NMAE normalisisation
    
    Returns
    -------
    MSE: ndarray
        MSE loss for each predictionstep
    NMAE : ndarray
        NMAE loss for each predicitonstep
    """
    
    tau_max = 12*predictlength
    model.eval()
    history_length = 12*intervallength
    n = len(idx_Power)
    epsilon = np.zeros((tau_max, n), dtype = 'float32')
    batches, remainder = np.divmod(n, batch_size)
    if remainder != 0:
        batches = batches+1
    if reg == True:
        if area<15:
            reg_test = reg_test[:,0].astype('float32')
        else:
            reg_test = reg_test[:,1].astype('float32')
        with torch.no_grad():
            for i in range(batches):
                print("Batch number {} of {}".format(i+1,batches))
                if i == batches-1:
                    batch_size = remainder
                for tau in range(tau_max):
                    data = np.zeros((batch_size, history_length+1), dtype = 'float32')
                    power = np.zeros((batch_size, history_length), dtype = 'float32')
                    target = np.zeros(batch_size, dtype = 'float32')
                    regularisation = np.zeros(batch_size, dtype = 'float32')
                    for j in range(batch_size):
                        idx = idx_Power[j]
                        start = idx +tau
                        end = start + history_length
                        pred = end + 1
                        power[j, :] = IMF[start:end].astype('float32')
                        target[j] = IMF[pred].astype('float32')
                        regularisation[j] = reg_test[pred]
                    data[:, 0] = regularisation
                    data[:, 1:] = power
                    data, target = torch.from_numpy(data).type(torch.Tensor), torch.from_numpy(target).type(torch.Tensor)
                    data, target = data.to(device), target.to(device)
                    output = model(data).squeeze()
                    if i == batches-1:
                        epsilon[tau, -batch_size:] = torch.sum(output - target).cpu().numpy()
                    else:
                        epsilon[tau, i*batch_size: (i+1)*batch_size] = torch.sum(output - target).cpu().numpy()
    else:
        with torch.no_grad():
            for i in range(batches):
                print("Batch number {} of {}".format(i+1,batches))
                if i == batches-1:
                    batch_size = remainder
                for tau in range(tau_max):
                    data = np.zeros((batch_size, history_length), dtype = 'float32')
                    target = np.zeros(batch_size, dtype = 'float32')
                    for j in range(batch_size):
                        idx = idx_Power[j]
                        start = idx +tau
                        end = start + history_length
                        pred = end + 1
                        data[j, :] = IMF[start:end].astype('float32')
                        target[j] = IMF[pred].astype('float32')
                    data, target = torch.from_numpy(data).type(torch.Tensor), torch.from_numpy(target).type(torch.Tensor)
                    data, target = data.to(device), target.to(device)
                    output = model(data).squeeze()
                    if i == batches-1:
                        epsilon[tau, -batch_size:] = torch.sum(output - target).cpu().numpy()
                    else:
                        epsilon[tau, i*batch_size: (i+1)*batch_size] = torch.sum(output - target).cpu().numpy()
    return epsilon


def validation(model, device, valid_loader, batch_size):
    """

    Evaluates the validation set.
    
    Parameters
    ----------
    model : PyTorch model class
    device : device
        Options are torch cpu or cuda.
    valid_loader : Dataloader
        Dataloader for the validation set.
    batch_size : int
        Batch size used to parallelise the computations.

    Returns
    -------
    valid_loss : float
        The validation loss.
    eps : ndarray, size=(n_valid,)
        The residuals for the validation set.

    """
    
    model.eval()
    valid_loss = 0
    epsilon = np.zeros(len(valid_loader.dataset))
    batches, remainder = np.divmod(len(valid_loader.dataset), batch_size)
    with torch.no_grad():
        for idx, (valid, valid_target) in enumerate(valid_loader):
            valid, valid_target = valid.to(device), valid_target.to(device)
            output_valid = model(valid).squeeze()
            valid_loss += F.mse_loss(output_valid, valid_target, reduction='sum').item()  # sum up batch loss
            if idx == batches:
                epsilon[idx*batch_size:] = (output_valid - valid_target).cpu().numpy()
            else:
                epsilon[idx*batch_size: (idx+1)*batch_size] = (output_valid - valid_target).cpu().numpy()
    valid_loss /= len(valid_loader.dataset)
    model.train()
    return valid_loss, epsilon