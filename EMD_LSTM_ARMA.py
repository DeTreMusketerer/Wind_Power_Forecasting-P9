"""
Dette script er et eventyr
"""

import Parrallel_RNN as pr
from sVARMAX.sVARMAX_Module import sVARMAX_quick_fit
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import pandas as pd
import Import_Data as imp
from scipy.io import savemat


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
    MSE = np.zeros(tau_max, dtype = 'float32')
    epsilon = np.zeros(tau_max, dtype = 'float32')
    history_length = 12*intervallength
    l = len(idx_Power)
    k, probably = np.divmod(l, batch_size)
    if probably != 0:
        k = k+1
    if reg == True:
        if area<15:
            reg_test = reg_test[:,0].astype('float32')
        else:
            reg_test = reg_test[:,1].astype('float32')
        with torch.no_grad():
            for i in range(k):
                print("Batch number {} of {}".format(i+1,k))
                if i == k-1:
                    batch_size = probably
                for tau in range(tau_max):
                    data = np.zeros((batch_size,history_length+1), dtype = 'float32')
                    power = np.zeros((batch_size,history_length), dtype = 'float32')
                    target = np.zeros(batch_size, dtype = 'float32')
                    regularisation = np.zeros(batch_size, dtype = 'float32')
                    for j in range(batch_size):
                        idx = idx_Power[j]
                        start = idx +tau
                        end = start + history_length
                        pred = end + 1
                        power[j,:] = IMF[start:end].astype('float32')
                        target[j] = IMF[pred].astype('float32')
                        regularisation[j] = reg_test[pred]
                    data[:,0] = regularisation
                    data[:,1:] = power
                    data, target = torch.from_numpy(data).type(torch.Tensor), torch.from_numpy(target).type(torch.Tensor)
                    data, target = data.to(device), target.to(device)
                    output = model(data).squeeze()
                    MSE[tau] += F.mse_loss(output, target, reduction='sum').item()
                    epsilon[tau] += torch.sum(torch.absolute(target - output)).item()
            MSE /= l
            epsilon /= l
    else:
        with torch.no_grad():
            for i in range(k):
                print("Batch number {} of {}".format(i+1,k))
                if i == k-1:
                    batch_size = probably
                for tau in range(tau_max):
                    data = np.zeros((batch_size,history_length), dtype = 'float32')
                    target = np.zeros(batch_size, dtype = 'float32')
                    for j in range(batch_size):
                        idx = idx_Power[j]
                        start = idx +tau
                        end = start + history_length
                        pred = end + 1
                        data[j,:] = IMF[start:end].astype('float32')
                        target[j] = IMF[pred].astype('float32')
                    data, target = torch.from_numpy(data).type(torch.Tensor), torch.from_numpy(target).type(torch.Tensor)
                    data, target = data.to(device), target.to(device)
                    output = model(data).squeeze()
                    MSE[tau] += F.mse_loss(output, target, reduction='sum').item()
                    epsilon[tau] += torch.sum(torch.absolute(target - output)).item()
            MSE /= l
            epsilon /= l
    return MSE, epsilon


def test_new(model, device, predictlength, IMF, reg_test, batch_size, area, intervallength, idx_Power, reg = True):
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



file_train = scipy.io.loadmat('data_energinet/New_Training_Data.mat')
file_test = scipy.io.loadmat('data_energinet/New_Test_Data.mat')
file_subtrain = scipy.io.loadmat('data_energinet/New_Subtraining_Data.mat')
file_valid = scipy.io.loadmat('data_energinet/New_Validation_Data.mat')

# n_train, k = file_train["y"].shape
# n_test, _ = file_test["y"].shape
# n_subtrain, _ = file_subtrain["y"].shape
# n_validation, _ = file_valid["y"].shape

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

threshold = 1e-03
tau_ahead = 288
p = 1
d = 0
q = 0
p_s = 0
d_s = 0 
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
dropout_hidden = 0.2
gamma = 0.7
log_interval = 100
patience = 15

intervallength = 1
predictlength = 24
tau_ahead = 288

idx_Power_train = imp.Index_dict_train_Power(intervallength)
idx_Power_val = imp.Index_dict_validation_Power(intervallength)
idx_Power_sub = imp.Index_dict_subtrain_Power(intervallength)
idx_Power_test = imp.Index_dict_Power_Test(intervallength, predictlength)
wind_areas = ["DK1-1", "DK1-2", "DK1-3", "DK1-4", "DK1-5", "DK1-6", "DK1-7",
              "DK1-8", "DK1-9", "DK1-10", "DK1-11", "DK1-12", "DK1-13",
              "DK1-14", "DK1-15", "DK2-1", "DK2-2", "DK2-3", "DK2-4", "DK2-5",
              "DK2-6"]

Is_reg_true = False
if Is_reg_true == True:
    input_size = intervallength*12 + 1
    r_part = 2
else:
    input_size = intervallength*12
    r_part = 1
shift = intervallength*12 + 1

idx_array_Power_val = np.array(list(idx_Power_val.values()))
idx_array_Power_test = np.array(list(idx_Power_test.values()))

with open(f"Learning/specs.txt", "w") as f:
    f.write("Batch size {}\n".format(batch_size))
    f.write("Input_size {}\n".format(input_size))
    f.write("Hidden sizes {}, {}\n".format(hidden_size, hidden_size2))
    f.write("Dropout {}\n".format(dropout_hidden))
    f.write("Hours history used {}\n".format(intervallength))
    f.write("gamma {}\n".format(gamma))
    f.write("logging interval {}, patience {}\n".format(log_interval, patience))
    f.write("Learning rate {}\n".format(learning_rate))
    f.write("SampEN threshold {}\n".format(threshold))
    f.write("Autoregressive order {}\n".format(p))
    f.write("Reg is {}".format(Is_reg_true))

for area_idx, wind_area in enumerate(wind_areas):
    print("Area {} of {}".format(area_idx+1, 21))
    sample_entropy = np.load(f'data_energinet/EMD_Power/{wind_area}_SampEN.npy')[:, -1]
    epsilons = np.zeros((len(idx_Power_val), 22))
    epsilons_test = np.zeros((tau_ahead, len(idx_Power_test), 22))
    for idx, SampEN in enumerate(sample_entropy):
        IMF_train = IMFs_train[:,area_idx,idx]
        IMF_subtrain = IMFs_sub[:,area_idx,idx]
        IMF_validation = IMFs_val[:,area_idx,idx]
        IMF_test = IMFs_test[:,area_idx,idx]
        IMF_max = np.max(IMF_subtrain)

        if SampEN < threshold: #time series
            Training_data = {"y": np.expand_dims(IMF_subtrain, -1), "z_reg": reg_sub, "z_NWP": z_sub, "missing_t": missing_t_sub}
            mod = sVARMAX_quick_fit(p, d, q, p_s, q_s, s, m, m_s,
                                    Training_data, r_part=r_part, l=area_idx, EMD=True, reg = Is_reg_true)
            mod.fit(reg = Is_reg_true)
            Validation_data = {"y": np.expand_dims(IMF_validation, -1), "z_reg": reg_val, "missing_t": missing_t_val}
            Phi, Psi, Xi, Sigma_u = mod.return_parameters()
            _, epsilon = mod.EMD_test(1, Validation_data, reg = Is_reg_true)
            epsilons[:, idx] = epsilon[0, idx_array_Power_val+shift, 0]
            
        else: # Neural network
            dset1 = PyTorchDataset(intervallength, IMF_train, reg_train, area_idx, idx_Power_train, reg = Is_reg_true)
            train_loader = torch.utils.data.DataLoader(dset1, batch_size, shuffle = False)
            dset3 = PyTorchDataset(intervallength, IMF_subtrain, reg_sub, area_idx, idx_Power_sub, reg = Is_reg_true)
            subtrain_loader = torch.utils.data.DataLoader(dset3, batch_size, shuffle = False)
            dset4 = PyTorchDataset(intervallength, IMF_validation, reg_val, area_idx, idx_Power_val, reg = Is_reg_true)
            valid_loader = torch.utils.data.DataLoader(dset4, batch_size, shuffle = False)
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = pr.LSTM(input_size, hidden_size, hidden_size2, dropout_hidden).to(device)    
            optimiser = optim.RMSprop(model.parameters(), lr = learning_rate)
            scheduler = StepLR(optimiser, step_size=1, gamma = gamma)

            opt_upd, upd_epoch, valid_loss, training_loss, min_valid_loss = pr.early_stopping(
                model, device, optimiser, scheduler, subtrain_loader, valid_loader, log_interval, patience, epochs)
            
            model_name = 'LSTM_N1{}_N2{}_batchsize{}_gamma{}_learningrate{}_Dropout{}_area{}_IMF{}'.format(
                hidden_size, hidden_size2, batch_size, gamma, learning_rate, dropout_hidden, area_idx, idx+1)

            dictionary = {'Validation_loss': valid_loss, 'Training_loss': training_loss}  
            df3 = pd.DataFrame(dictionary) 
            df3.to_csv('Learning/Train_' + model_name + '.csv', index=False)
            
            _, epsilon_array = pr.validation_new(model, device, valid_loader, batch_size)
            epsilons[:,idx] = epsilon_array

        # RE-TRAIN
        if SampEN < threshold: #time series
            Training_data = {"y": np.expand_dims(IMF_train, -1), "z_reg": reg_train, "z_NWP": z_train, "missing_t": missing_t_train}
            mod = sVARMAX_quick_fit(p, d, q, p_s, q_s, s, m, m_s,
                                    Training_data, r_part=r_part, l=area_idx, EMD=True, reg = Is_reg_true)
            mod.fit(reg = Is_reg_true)
            Test_data = {"y": np.expand_dims(IMF_test, -1), "z_reg": reg_test, "missing_t": missing_t_test}
            Phi, Psi, Xi, Sigma_u = mod.return_parameters()
            _, epsilon = mod.EMD_test(tau_ahead, Test_data, reg = Is_reg_true)
            epsilons_test[:, :, idx] = epsilon[:, idx_array_Power_test+shift, 0]
            save_dict = {"Xi": Xi, "Sigma_u": Sigma_u}
            if p != 0 or q_s != 0:
                save_dict.update({"Phi": Phi})
            if q != 0 or q_s != 0:
                save_dict.update({"Psi": Psi})
            savemat(f"models/ARMA_IMF{idx+1}_{wind_area}.mat", save_dict)

        else:
            print(f'\n Re-training:\n')
            model = pr.LSTM(input_size, hidden_size, hidden_size2, dropout_hidden).to(device) # Reset model parameters
            optimiser = optim.RMSprop(model.parameters(), lr = learning_rate)
            scheduler = StepLR(optimiser, step_size=1, gamma = gamma)

            updates_counter = 0
            epoch = 1
            while updates_counter < opt_upd:
                updates_counter = pr.early_stopping_retrain(model, device, train_loader,
                                                          optimiser, epoch, opt_upd,
                                                          updates_counter, scheduler,
                                                          upd_epoch, log_interval)
                print('')
                epoch += 1

            save_model = True
            if save_model is True:
                    torch.save(model.state_dict(), 'models/' + model_name + '.pt')

            epsilon_array = test_new(model, device, predictlength, IMF_test, reg_test, batch_size,
                             area_idx, intervallength, idx_Power_test, reg = Is_reg_true)
            epsilons_test[:, :, idx] = epsilon_array

    np.save(f"Learning/EMD_Test_{wind_area}.npy", epsilons_test)
    np.save(f"Learning/EMD_Validation_{wind_area}.npy", epsilons)
