"""
Parrallel-RNN on the danish wind power production.
"""


import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch import optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import scipy.io
import pandas as pd
import Import_Data as imp


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
    def __init__(self, area, intervallength, y, z, reg, idx_power, idx_nwp):
        self.area = area
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
        external = self.z[index_n[0],index_n[1],self.area*11:(self.area+1)*11].astype('float32')
        power = self.y[index_p:index_p+self.endpoint, self.area].astype('float32')
        if self.area<15:
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


def test_new(model, device, predictlength, z, y, reg, batch_size, area, intervallength, idx_Power, idx_NWP, y_max):
    """
    Evaluates the neural network in terms of MSE and NMAE.
    
    Parameters
    ----------
    model : PyTorch model class
    device : device
    predictionlength : int
        Number of hours we predict ahead
    z : ndarray
        NWPs data
    y : ndarray
        Power data
    reg : ndarray
        Regularisation data
    batch_size : int
        Batch size
    area : int in [0-20]
        integer telling us which wind area we are in 
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
    NMAE = np.zeros(tau_max, dtype = 'float32')
    point = 12*intervallength
    n = 12*intervallength+12
    l = len(idx_Power)
    k = len(idx_NWP)
    if area<15:
        reg = reg[:,0].astype('float32')
    else:
        reg = reg[:,1].astype('float32')
    with torch.no_grad():
        for i in range(k):
            a = idx_NWP[i]
            print("Batch number {} of {}".format(i+1,k))
            if i == k-1:
                batch_size = len(idx_NWP_test[k-1])
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
                    Power_batch[j,:] = y[start:end,area].astype('float32')
                    target[j] = y[pred,area].astype('float32')
                    regularisation[j] = reg[pred].astype('float32')
                external = z[a[:,0]+(tau//12),a[:,1],area*11:(area+1)*11].astype('float32')
                data[:,0] = regularisation
                data[:,1:12] = external
                data[:,12:] = Power_batch
                data, target = torch.from_numpy(data).type(torch.Tensor), torch.from_numpy(target).type(torch.Tensor)
                data, target = data.to(device), target.to(device)
                output = model(data).squeeze()
                MSE[tau] += F.mse_loss(output, target, reduction='sum').item()
                NMAE[tau] += torch.sum(torch.absolute(target - output)/y_max).item()
        MSE /= l
        NMAE /= l
    return MSE, NMAE


def validation(model, device, valid_loader):
    """
    Evaluates the validation set. This is done during early stopping.
    
    Parameters
    ----------
    model : PyTorch model class
    device : device
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


def validation_new(model, device, valid_loader, batch_size):
    """
    Evaluates the validation set. This is done during early stopping.
    
    Parameters
    ----------
    model : PyTorch model class
    device : device
    valid_loader : Dataloader
        Dataloader for the validation set.
    batch_size : int
        Batch size
    Returns
    -------
    valid_loss : float
        The validation loss.
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


def early_stopping(model, device, optimiser, scheduler, subtrain_loader,
                    valid_loader, log_interval, patience, epochs):
    """
    Determines the number of parameter updates which should be performed
    during training using early stopping.
    
    Parameters
    ----------
    model : PyTorch model class
    device : device
    optimiser : PyTorch optimiser
    scheduler : PyTorch scheduler
    subtrain_loader : Dataloder
        Subtrain dataset.
    valid_loader : Dataloader
        Validation dataset.
    log_interval: int
        Time between prints of performance.
    patience : int
        Patience parameter in early stopping
    epochs : int
        Max number of epochs to train
    Returns
    -------
    optim_updates : int
        The optimal number of parameter updates during training
        as determined by early stopping.
    updates_pr_pretrain_epoch : int
        Number of paramer updates per epoch in the subtrain dataset.
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
                valid_loss = validation(model, device, valid_loader)
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
        time between prints of performance.
        
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


def early_stopping_retrain_new(model, device, train_loader, optimiser, epoch,
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
        time between prints of performance.
        
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

if __name__ == '__main__':
    # Parameters
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    epochs = 100
    input_size = 588
    batch_size = 32
    learning_rate = 0.00011
    hidden_size = 512
    hidden_size2 = 512
    hidden_size3 = 512
    dropout_hidden = 0.1
    gamma = 0.7
    log_interval = 100
    patience = 100
    
    # Data
    area_list = [i for i in range(21)]
    area = area_list[0]
    intervallength = 48 # Number of days in a training datapoint.
    predictlength = 24 # Number of hours we predict ahead.
    
    file_train = scipy.io.loadmat('data_energinet/New_Training_Data.mat')
    file_test = scipy.io.loadmat('data_energinet/New_Test_Data.mat')
    file_subtrain = scipy.io.loadmat('data_energinet/New_Subtraining_Data.mat')
    file_valid = scipy.io.loadmat('data_energinet/New_Validation_Data.mat')
    
    y_train = np.float32(file_train['y'])
    z_train = np.float32(file_train['z_NWP'])
    reg_train = np.float32(file_train['z_reg'])
    
    y_test = np.float32(file_test['y'])[24:, :]
    z_test = np.float32(file_test['z_NWP'])
    reg_test = np.float32(file_test['z_reg'])[24:, :]
    
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
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    
    all_areas = True
    model_start = GRU(input_size, hidden_size, hidden_size2, hidden_size3, dropout_hidden).to(device)
    Type = "GRU"
    
    if all_areas == True:
        for area in area_list:
            dset1 = PyTorchDataset(area, intervallength, y_train, z_train, reg_train, idx_Power_train, idx_NWP_train)
            train_loader = torch.utils.data.DataLoader(dset1, batch_size, shuffle = False)
            dset3 = PyTorchDataset(area, intervallength, y_sub, z_sub, reg_sub, idx_Power_sub, idx_NWP_sub)
            subtrain_loader = torch.utils.data.DataLoader(dset3, batch_size, shuffle = False)
            dset4 = PyTorchDataset(area, intervallength, y_val, z_val, reg_val, idx_Power_val, idx_NWP_val)
            valid_loader = torch.utils.data.DataLoader(dset4, batch_size, shuffle = False)
            
            model = model_start
            
            # Load a previously trained model?
            model_name = '{}_N1{}_N2{}_gamma{}_learningrate{}_Dropout{}_intervallength{}_area{}'.format(Type,
                hidden_size, hidden_size2, batch_size, gamma, learning_rate, dropout_hidden, intervallength, area)
            load_model = False
            if load_model is True:
                    model.load_state_dict(torch.load('models/' + model_name + '.pt', map_location=device))
            
            # Training 
            optimiser = optim.RMSprop(model.parameters(), lr = learning_rate)
            scheduler = StepLR(optimiser, step_size=1, gamma = gamma)
            
            opt_upd, upd_epoch, valid_loss, training_loss, min_valid_loss = early_stopping(
                model, device, optimiser, scheduler, subtrain_loader, valid_loader, log_interval, patience, epochs)
            
            print('\n Re-training:\n')
            model = model_start # Reset model parameters
            optimiser = optim.RMSprop(model.parameters(), lr = learning_rate)
            scheduler = StepLR(optimiser, step_size=1, gamma = gamma)
            
            updates_counter = 0
            epoch = 1
            while updates_counter < opt_upd:
                updates_counter = early_stopping_retrain(model, device, train_loader,
                                                          optimiser, epoch, opt_upd,
                                                          updates_counter, scheduler,
                                                          upd_epoch, log_interval)
                print('')
                epoch += 1
            
            print(min_valid_loss)
            
            #Save learing curves
            dictionary = {'Validation_loss': valid_loss, 'Training_loss': training_loss}  
            df3 = pd.DataFrame(dictionary) 
            df3.to_csv('Learning/Train_' + model_name + '.csv', index=False)
            
            # # Save model?
            save_model = True
            if save_model is True:
                    torch.save(model.state_dict(), 'models/' + model_name + '.pt')
            
            # Testing
            y_max = np.zeros(21, dtype = 'float32')
            for i in range(21):
                y_max[i] = np.amax(y_train[:,i])
            
            print('Testing')
            MSE, NMAE = test_new(model, device, predictlength, z_test, y_test, reg_test, batch_size, area,
                                  intervallength, idx_Power_test, idx_NWP_test, y_max[area])
            
            # Save results
            with open('Learning/Test_MSE_' + model_name + '.npy', 'wb') as file:
                np.save(file, MSE)
            with open('Learning/Test_NMAE_' + model_name + '.npy', 'wb') as file:
                np.save(file, NMAE)
    else:
        dset1 = PyTorchDataset(area, intervallength, y_train, z_train, reg_train, idx_Power_train, idx_NWP_train)
        train_loader = torch.utils.data.DataLoader(dset1, batch_size, shuffle = False)
        dset3 = PyTorchDataset(area, intervallength, y_sub, z_sub, reg_sub, idx_Power_sub, idx_NWP_sub)
        subtrain_loader = torch.utils.data.DataLoader(dset3, batch_size, shuffle = False)
        dset4 = PyTorchDataset(area, intervallength, y_val, z_val, reg_val, idx_Power_val, idx_NWP_val)
        valid_loader = torch.utils.data.DataLoader(dset4, batch_size, shuffle = False)

        model = model_start
        
        # Load a previously trained model?
        model_name = '{}_N1{}_N2{}_batchsize{}_gamma{}_learningrate{}_Dropout{}_intervallength_{}_area{}'.format(Type,
            hidden_size, hidden_size2, batch_size, gamma, learning_rate, dropout_hidden, intervallength, area)
        load_model = False
        if load_model is True:
                model.load_state_dict(torch.load('models/' + model_name + '.pt', map_location=device))
        
        # Training 
        optimiser = optim.RMSprop(model.parameters(), lr = learning_rate)
        scheduler = StepLR(optimiser, step_size=1, gamma = gamma)
        
        opt_upd, upd_epoch, valid_loss, training_loss, min_valid_loss = early_stopping(
            model, device, optimiser, scheduler, subtrain_loader, valid_loader, log_interval, patience, epochs)
        
        # print('\n Re-training:\n')
        # model = model_start # Reset model parameters
        # optimiser = optim.RMSprop(model.parameters(), lr = learning_rate)
        # scheduler = StepLR(optimiser, step_size=1, gamma = gamma)
        
        # updates_counter = 0
        # epoch = 1
        # while updates_counter < opt_upd:
        #     updates_counter = early_stopping_retrain(model, device, train_loader,
        #                                               optimiser, epoch, opt_upd,
        #                                               updates_counter, scheduler,
        #                                               upd_epoch, log_interval)
        #     print('')
        #     epoch += 1
        
        # print(min_valid_loss)
        
        # #Save learing curves
        # dictionary = {'Validation_loss': valid_loss, 'Training_loss': training_loss}  
        # df3 = pd.DataFrame(dictionary) 
        # df3.to_csv('Learning/Train_' + model_name + '.csv', index=False)
        
        # # Save model?
        # save_model = True
        # if save_model is True:
        #         torch.save(model.state_dict(), 'models/' + model_name + '.pt')
        
        # Testing
        # y_max = np.zeros(21, dtype = 'float32')
        # for i in range(21):
        #     y_max[i] = np.amax(y_train[:,i])
        
        # print('Testing')
        # MSE, NMAE = test_new(model, device, predictlength, z_test, y_test, reg_test, batch_size, area, intervallength, idx_Power_test, idx_NWP_test, y_max[area])
        
        # # Save results
        # with open('Learning/Test_MSE_' + model_name + '.npy', 'wb') as file:
        #     np.save(file, MSE)
        # with open('Learning/Test_NMAE_' + model_name + '.npy', 'wb') as file:
        #     np.save(file, NMAE)