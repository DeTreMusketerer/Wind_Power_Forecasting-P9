# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 09:25:25 2021


Authors:  Andreas Anton Andersen, Martin Voigt Vejling, and Morten Stig Kaaber
E-Mails: {aand17, mvejli17, mkaabe17}@student.aau.dk

Contains functionality for the persistence method used on the Danish 
wind power production. 
See the report
        Forecasting Wind Power Production
            - Chapter 6: Experimental Setup
                - Section 6.2.1: Persistence
                
The script has been developed using Python 3.9 with the
libraries numpy, scipy, inspect, and os.
"""

import numpy as np
import datetime as dt
from scipy.io import loadmat
import os
import inspect




def persistence(train_data, test_data, t_start, l, tau_ahead):
    """
    Persistence method for a tau-ahead prediction.
    
    Parameters
    ----------
    train_data : Wind power production data in training set

    test_data : Wind power production data in test set
    
    t_start : int
        Time index for which the prediction shall start 
    
    l : int
        Sub-grid index starting from 0 to 20
        
    tau_ahead : int
        Prediction length 
    
    Returns
    -------
    tau_ahead_forecast: ndarray
        Tau-ahead forecast using the persistence method for sub-grid l

    """
    
    train_power_his = train_data["y"][:,l]
    test_power_his = test_data["y"][:,l]
    tau_ahead_forecast = np.zeros((len(test_power_his),tau_ahead))
    i = 0
    for t in range(t_start,len(test_power_his)):
        for tau in range(tau_ahead):
            if t_start+t-1<0:
                tau_ahead_forecast[t,tau] = train_power_his[-1]
            else:
                tau_ahead_forecast[i,tau] = test_power_his[t-1]
        i = i + 1
    return tau_ahead_forecast


def Evaluation(train_data, test_data, missing_t, t_start, tau_ahead):
    """
    Persistence method for a tau-ahead prediction.
    
    Parameters
    ----------
    train_data : Wind power production data in training set

    test_data : Wind power production data in test set
    
    missing_t : Time indices for which the wind power production is missing
    
    t_start : int
        Time index for which the prediction shall start 
        
    tau_ahead : int
        Prediction length 
    
    Returns
    -------
    MSE: ndarray
        MSE loss for each prediction step and each sub-grid
    NMAE : ndarray
        NMAE loss for each prediciton step and each sub-grid

    """
    
    MSE_matrix = np.zeros((21,tau_ahead))
    NMAE_matrix = np.zeros((21,tau_ahead))
    for l in range(21):
        idx_list = []
        forecast = persistence(train_data, test_data, t_start, l, tau_ahead)
        test = test_data["y"][:,l]
        P_max = np.max(train_data["y"][:,l])
        for miss_idx in range(len(missing_t)-1):
            for t in range(missing_t[miss_idx]+1, missing_t[miss_idx+1]-tau_ahead):
                idx_list.append(t)
                
        eps = np.zeros((len(idx_list), tau_ahead))
        idx_list = np.array(idx_list)
        for tau in range(tau_ahead):
            eps[:,tau] = forecast[idx_list,tau] - test[idx_list+tau]
        MSE = np.mean(eps**2, axis=0)
        NMAE = np.mean(np.abs(eps), axis=0)/P_max
        MSE_matrix[l,:] = MSE
        NMAE_matrix[l,:] = NMAE
    return MSE_matrix, NMAE_matrix


if __name__ == '__main__':
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    
    
    Train_TS = loadmat(currentdir+"/data_energinet/Training_data_TS.mat")
    Test_TS = loadmat(currentdir+"/data_energinet/Test_data_TS.mat")
    
    tau_ahead = 1
    
    test = Test_TS["y"]
    missing_t = Test_TS["missing_t"][0]

    eva = Evaluation(Train_TS, Test_TS, missing_t, 0, tau_ahead)
    mse = eva[0] 
    nmae = eva[1]

    
    average_MSE = np.zeros(tau_ahead)
    for tau in range(tau_ahead):
        average_MSE[tau] = np.mean(mse[:,tau])
        
    average_NMAE = np.zeros(tau_ahead)
    for tau in range(tau_ahead):
        average_NMAE[tau] = np.mean(nmae[:,tau])
    
    