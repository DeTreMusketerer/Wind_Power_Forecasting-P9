# -*- coding: utf-8 -*-
"""

Created on Mon Nov 29 08:54:02 2021

Authors:  Andreas Anton Andersen, Martin Voigt Vejling, and Morten Stig Kaaber
E-Mails: {aand17, mvejli17, mkaabe17}@student.aau.dk

In this script used to train, test, and validate the EMD-LSTM-ARMA
model is given, see the report
        Forecasting Wind Power Production
            - Chapter 6: Experimental Setup
                - Section 6.2.6: EMD-LSTM-ARMA

The script has been developed using Python 3.9 with the
libraries numpy, scipy, pandas, and pytorch.

"""

import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from scipy.io import savemat, loadmat

import Modules.Parrallel_RNN_Module as U_RNN
from Modules.sVARMAX_Module import sVARMAX
import Import_Data as imp
from Modules.EMD_RNN import PyTorchDataset, test, validation


if __name__ == "__main__":
    model_basename = "EMD_LSTM_ARMA_001"
    use_reg = False # Use regulation data?
    save_model = True

    # Parameters
    threshold = 1e-03
    p = 1
    q = 0
    p_s = 0
    q_s = 0
    s = 0
    m = 0
    m_s = 0

    # Neural Network
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    epochs = 100
    batch_size = 64
    learning_rate = 2e-04
    hidden_size = 512
    hidden_size2 = 512
    hidden_size3 = 512
    dropout_hidden = 0.2
    gamma = 0.7
    log_interval = 100
    patience = 15
    
    intervallength = 1 # Number of hours in a training datapoint.
    predictlength = 24 # Number of hours we predict ahead.
    tau_ahead = 288

    file_train = loadmat('data_energinet/New_Training_Data.mat')
    file_test = loadmat('data_energinet/New_Test_Data.mat')
    file_subtrain = loadmat('data_energinet/New_Subtraining_Data.mat')
    file_valid = loadmat('data_energinet/New_Validation_Data.mat')

    IMFs_train = np.load('data_energinet/EMD_Training_Data.npy')
    z_train = np.float32(file_train['z_NWP'])
    reg_train = np.float32(file_train['z_reg'])
    missing_t_train = file_train['missing_t'][0, :]
    
    IMFs_test = np.load('data_energinet/EMD_Test_Data.npy')[24:, :, :]
    z_test = np.float32(file_test['z_NWP'])
    reg_test = np.float32(file_test['z_reg'])[24:, :]
    missing_t_test = file_test['missing_t'][0, :]
    missing_t_test[1:] = missing_t_test[1:]-24
    
    IMFs_sub = np.load('data_energinet/EMD_Subtraining_Data.npy')
    z_sub = np.float32(file_subtrain['z_NWP'])
    reg_sub = np.float32(file_subtrain['z_reg'])
    missing_t_sub = file_subtrain['missing_t'][0, :]
    
    IMFs_val = np.load('data_energinet/EMD_Validation_Data.npy')
    z_val = np.float32(file_valid['z_NWP'])
    reg_val = np.float32(file_valid['z_reg'])
    missing_t_val = file_valid['missing_t'][0, :]
    
    idx_Power_train = imp.Index_dict_train_Power(intervallength)
    idx_Power_val = imp.Index_dict_validation_Power(intervallength)
    idx_Power_sub = imp.Index_dict_subtrain_Power(intervallength)
    idx_Power_test = imp.Index_dict_Power_Test(intervallength, predictlength)
    wind_areas = ["DK1-1", "DK1-2", "DK1-3", "DK1-4", "DK1-5", "DK1-6", "DK1-7",
                  "DK1-8", "DK1-9", "DK1-10", "DK1-11", "DK1-12", "DK1-13",
                  "DK1-14", "DK1-15", "DK2-1", "DK2-2", "DK2-3", "DK2-4", "DK2-5",
                  "DK2-6"]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if Is_reg_true == True:
        input_size = intervallength*12 + 1
        r_part = 2
    else:
        input_size = intervallength*12
        r_part = 1
    shift = intervallength*12 + 1
    
    idx_array_Power_val = np.array(list(idx_Power_val.values()))
    idx_array_Power_test = np.array(list(idx_Power_test.values()))
    
    with open("Models/specs.txt", "w") as f:
        f.write("Batch size {}\n".format(batch_size))
        f.write("Input_size {}\n".format(input_size))
        f.write("Hidden sizes {}, {}\n".format(hidden_size, hidden_size2))
        f.write("Dropout {}\n".format(dropout_hidden))
        f.write("Hours history used {}\n".format(intervallength))
        f.write("logging interval {}, patience {}\n".format(log_interval, patience))
        f.write("Learning rate {}\n".format(learning_rate))
        f.write("Learning rate decay {}\n".format(gamma))
        f.write("SampEN threshold {}\n".format(threshold))
        f.write("Autoregressive order {}\n".format(p))
        f.write("Reg is {}".format(use_reg))
    
    for area_idx, wind_area in enumerate(wind_areas):
        print("Area {} of {}".format(area_idx+1, 21))
        sample_entropy = np.load(f'data_energinet/EMD_Power/{wind_area}_SampEN.npy')[:, -1]
        epsilons = np.zeros((len(idx_Power_val), 22))
        epsilons_test = np.zeros((tau_ahead, len(idx_Power_test), 22))
        for idx, SampEN in enumerate(sample_entropy):
            model_name = f"{model_basename}_{wind_area}_IMF{idx+1}"

            IMF_train = IMFs_train[:,area_idx,idx]
            IMF_subtrain = IMFs_sub[:,area_idx,idx]
            IMF_validation = IMFs_val[:,area_idx,idx]
            IMF_test = IMFs_test[:,area_idx,idx]
            IMF_max = np.max(IMF_subtrain)

            if SampEN < threshold: #time series
                mod = sVARMAX(np.expand_dims(IMF_subtrain, -1), reg_sub, z_sub, missing_t_sub,
                              p=p, d=0, q=q, p_s=p_s, q_s=q_s, s=s, m=m, m_s=m_s, l=area_idx,
                              use_NWP=False, use_reg=use_reg)
                mod.fit()
                Phi, Psi, Xi, Sigma_u = mod.return_parameters()
                _, _, epsilon = mod.test(1, np.expand_dims(IMF_validation, -1), reg_val, z_val,
                                         missing_t_val, P_max=np.ones(21))
                epsilons[:, idx] = epsilon[0, idx_array_Power_val+shift, 0]

            else: # Neural network
                dset1 = PyTorchDataset(intervallength, IMF_train, reg_train, area_idx, idx_Power_train, reg = use_reg)
                train_loader = torch.utils.data.DataLoader(dset1, batch_size, shuffle = False)
                dset3 = PyTorchDataset(intervallength, IMF_subtrain, reg_sub, area_idx, idx_Power_sub, reg = use_reg)
                subtrain_loader = torch.utils.data.DataLoader(dset3, batch_size, shuffle = False)
                dset4 = PyTorchDataset(intervallength, IMF_validation, reg_val, area_idx, idx_Power_val, reg = use_reg)
                valid_loader = torch.utils.data.DataLoader(dset4, batch_size, shuffle = False)

                model = U_RNN.LSTM(input_size, hidden_size, hidden_size2, hidden_size3, dropout_hidden).to(device)    
                optimiser = optim.RMSprop(model.parameters(), lr = learning_rate)
                scheduler = StepLR(optimiser, step_size=1, gamma = gamma)

                opt_upd, upd_epoch, valid_loss, training_loss, min_valid_loss = U_RNN.early_stopping(
                    model, device, optimiser, scheduler, subtrain_loader, valid_loader, log_interval, patience, epochs)
    
                dictionary = {'Validation_loss': valid_loss, 'Training_loss': training_loss}  
                df3 = pd.DataFrame(dictionary) 
                df3.to_csv(f"Learning/Train_{model_name}.csv", index=False)
                
                _, epsilon_array = validation(model, device, valid_loader, batch_size)
                epsilons[:,idx] = epsilon_array
    
            # RE-TRAIN
            if SampEN < threshold: #time series
                mod = sVARMAX(np.expand_dims(IMF_train, -1), reg_train, z_train, missing_t_train,
                              p=p, d=0, q=q, p_s=p_s, q_s=q_s, s=s, m=m, m_s=m_s, l=area_idx,
                              use_NWP=False, use_reg=use_reg)
                mod.fit()
                Phi, Psi, Xi, Sigma_u = mod.return_parameters()
                _, _, epsilon = mod.test(tau_ahead, np.expand_dims(IMF_test, -1), reg_test, z_test,
                                         missing_t_test, P_max=np.ones(21))
                epsilons_test[:, :, idx] = epsilon[:, idx_array_Power_test+shift, 0]
                save_dict = {"Xi": Xi, "Sigma_u": Sigma_u}
                if p != 0 or q_s != 0:
                    save_dict.update({"Phi": Phi})
                if q != 0 or q_s != 0:
                    save_dict.update({"Psi": Psi})
                savemat(f"models/ARMA_{wind_area}_IMF{idx+1}.mat", save_dict)
    
            else:
                print("\n Re-training:\n")
                model = U_RNN.LSTM(input_size, hidden_size, hidden_size2, hidden_size3, dropout_hidden).to(device) # Reset model parameters
                optimiser = optim.RMSprop(model.parameters(), lr = learning_rate)
                scheduler = StepLR(optimiser, step_size=1, gamma = gamma)
    
                updates_counter = 0
                epoch = 1
                while updates_counter < opt_upd:
                    updates_counter = U_RNN.early_stopping_retrain(model, device, train_loader,
                                                              optimiser, epoch, opt_upd,
                                                              updates_counter, scheduler,
                                                              upd_epoch, log_interval)
                    print('')
                    epoch += 1
    
                if save_model is True:
                        torch.save(model.state_dict(), 'models/' + model_name + '.pt')
    
                epsilon_array = test(model, device, predictlength, IMF_test, reg_test, batch_size,
                                     area_idx, intervallength, idx_Power_test, reg = use_reg)
                epsilons_test[:, :, idx] = epsilon_array
    
        np.save(f"Learning/EMD_Test_{wind_area}.npy", epsilons_test)
        np.save(f"Learning/EMD_Validation_{wind_area}.npy", epsilons)
