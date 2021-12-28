# -*- coding: utf-8 -*-
"""

Created on Thur Nov 18 14:33:46 2021

Authors:  Andreas Anton Andersen, Martin Voigt Vejling, and Morten Stig Kaaber
E-Mails: {aand17, mvejli17, mkaabe17}@student.aau.dk

Contains functionality used to train, test, and validate the multivariate
recurrent neural network models on the danish wind power production. See
the report
        Forecasting Wind Power Production
            - Chapter 5: Neural Networks
            - Chapter 6: Experimental Setup
                - Section 6.2.5: Multivariate RNN

The script has been developed using Python 3.9 with the
libraries numpy, pytorch, scipy, and pandas.

"""

import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
import scipy.io
import pandas as pd

import Modules.Import_Data as imp
import Modules.Multi_RNN_Module as M_RNN


if __name__ == "__main__":
    model_name = "M_RNN_001"
    Type = "GRU"
    all_areas = True
    load_model = False
    save_model = True

    # Parameters
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    input_size = 24
    epochs = 100 # maximum number of epochs (should not be relevant due to early stopping)
    batch_size = 128
    learning_rate = 0.00031
    hidden_size = 512
    hidden_size2 = 512
    hidden_size3 = 512
    dropout_hidden = 0.1
    eta_d = 0.7 # learning rate update
    num_layers = 1 # number of parrallel layers
    log_interval = 100
    patience = 100

    # Data
    intervallength = 1 # Number of hours in a training datapoint.
    predictlength = 24 # Number of hours we predict ahead.

    file_train = scipy.io.loadmat('data_energinet/New_Training_Data.mat')
    file_test = scipy.io.loadmat('data_energinet/New_Test_Data.mat')
    file_subtrain = scipy.io.loadmat('data_energinet/New_Subtraining_Data.mat')
    file_valid = scipy.io.loadmat('data_energinet/New_Validation_Data.mat')

    y_train = np.float32(file_train['y'])
    z_train = np.float32(file_train['z_NWP'])
    reg_train = np.float32(file_train['z_reg'])

    y_test = np.float32(file_test['y'])
    z_test = np.float32(file_test['z_NWP'])
    reg_test = np.float32(file_test['z_reg'])

    y_sub = np.float32(file_subtrain['y'])
    z_sub = np.float32(file_subtrain['z_NWP'])
    reg_sub = np.float32(file_subtrain['z_reg'])

    y_val = np.float32(file_valid['y'])
    z_val = np.float32(file_valid['z_NWP'])
    reg_val = np.float32(file_valid['z_reg'])

    idx_NWP_train = imp.Index_dict_train_NWP(intervallength)
    idx_NWP_val = imp.Index_dict_validation_NWP(intervallength)
    idx_NWP_sub = imp.Index_dict_subtrain_NWP(intervallength)
    idx_NWP_test = imp.Index_dict_NWP_Test(intervallength, predictlength, batch_size)

    idx_Power_train = imp.Index_dict_train_Power(intervallength)
    idx_Power_val = imp.Index_dict_validation_Power(intervallength)
    idx_Power_sub = imp.Index_dict_subtrain_Power(intervallength)
    idx_Power_test = imp.Index_dict_Power_Test(intervallength, predictlength)

    dset1 = M_RNN.PyTorchDataset(intervallength, y_train, z_train, reg_train, idx_Power_train, idx_NWP_train)
    train_loader = torch.utils.data.DataLoader(dset1, batch_size, shuffle = False)
    dset3 = M_RNN.PyTorchDataset(intervallength, y_sub, z_sub, reg_sub, idx_Power_sub, idx_NWP_sub)
    subtrain_loader = torch.utils.data.DataLoader(dset3, batch_size, shuffle = False)
    dset4 = M_RNN.PyTorchDataset(intervallength, y_val, z_val, reg_val, idx_Power_val, idx_NWP_val)
    valid_loader = torch.utils.data.DataLoader(dset4, batch_size, shuffle = False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    if Type == "GRU":
        model_start = M_RNN.GRU2(input_size, hidden_size, hidden_size2, hidden_size3, dropout_hidden, device).to(device)
    elif Type == "LSTM":
        model_start = M_RNN.LSTM2(input_size, hidden_size, hidden_size2, hidden_size3, dropout_hidden, device).to(device)

    wind_areas = ["DK1-1", "DK1-2", "DK1-3", "DK1-4", "DK1-5", "DK1-6", "DK1-7", "DK1-8", "DK1-9", "DK1-10", "DK1-11", "DK1-12", "DK1-13", "DK1-14", "DK1-15", "DK2-1", "DK2-2", "DK2-3", "DK2-4", "DK2-5", "DK2-6"]

    model = model_start

    with open(f"Models/specs_{model_name}.txt", "w") as f:
        f.write("Type: {}".format(Type))
        f.write("Batch size {}\n".format(batch_size))
        f.write("Input_size {}\n".format(input_size))
        f.write("Hidden sizes {}, {}\n".format(hidden_size, hidden_size2))
        f.write("Dropout {}\n".format(dropout_hidden))
        f.write("Hours history used {}\n".format(intervallength))
        f.write("logging interval {}, patience {}\n".format(log_interval, patience))
        f.write("Learning rate {}\n".format(learning_rate))
        f.write("Learning rate decay {}\n".format(eta_d))

    # Use a previously trained model?    
    if load_model is True:
        model.load_state_dict(torch.load('models/' + model_name + '.pt', map_location=device))
    else:
        # Training
        optimiser = optim.RMSprop(model.parameters(), lr = learning_rate)
        scheduler = StepLR(optimiser, step_size=1, gamma = eta_d)

        opt_upd, upd_epoch, valid_loss, training_loss, min_valid_loss = M_RNN.early_stopping(
            model, device, optimiser, scheduler, subtrain_loader, valid_loader, log_interval, patience, epochs)

        # Save learing curves
        dictionary = {'Validation_loss': valid_loss, 'Training_loss': training_loss}  
        df3 = pd.DataFrame(dictionary) 
        df3.to_csv('Learning/Train_' + model_name + '.csv', index=False)

        print('\n Re-training:\n')
        model = model_start # Reset model parameters
        optimiser = optim.RMSprop(model.parameters(), lr = learning_rate)
        scheduler = StepLR(optimiser, step_size=1, gamma = eta_d)

        updates_counter = 0
        epoch = 1
        while updates_counter < opt_upd:
            updates_counter = M_RNN.early_stopping_retrain(model, device, train_loader,
                                                           optimiser, epoch, opt_upd,
                                                           updates_counter, scheduler,
                                                           upd_epoch, log_interval)
            print('')
            epoch += 1

        # Save learing curves
        dictionary = {'Validation_loss': valid_loss, 'Training_loss': training_loss}  
        df3 = pd.DataFrame(dictionary) 
        df3.to_csv('Learning/Train_' + model_name + '.csv', index=False)

        # Save model?
        if save_model is True:
            torch.save(model.state_dict(), f"Models/{model_name}.pt")

    # Testing
    y_max = np.zeros(21)
    for i in range(21):
        y_max[i] = np.amax(y_train[:,i])
    
    print('Testing')
    MSE, NMAE = M_RNN.test(model, device, predictlength, z_test, y_test, reg_test,
                           batch_size, intervallength, idx_Power_test, idx_NWP_test, y_max)

    # Save results
    with open('Learning/Test_MSE_' + model_name + '.npy', 'wb') as file:
        np.save(file, MSE)
    with open('Learning/Test_NMAE_' + model_name + '.npy', 'wb') as file:
        np.save(file, NMAE)
