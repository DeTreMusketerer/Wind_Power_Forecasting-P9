# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 10:30:32 2021

@author: marti
"""

import os
import inspect

import numpy as np
from scipy.io import loadmat, savemat

from sVARMAX_Module import sVARMAX_quick_fit


if __name__ == '__main__':
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)

    np.random.seed(49)

    Training_data = loadmat(parentdir+"/data_energinet/New_Training_Data.mat")
    train_y = Training_data["y"]
    train_z_NWP = Training_data["z_NWP"]
    train_z_reg = Training_data["z_reg"]
    missing_t_train = Training_data["missing_t"][0, :]

    Test_data = loadmat(parentdir+"/data_energinet/New_Test_Data.mat")
    test_y = Test_data["y"][24:, :] # Need to start at 12:00:00 to match NWP
    test_z_NWP = Test_data["z_NWP"]
    test_z_reg = Test_data["z_reg"][24:, :]
    missing_t_test = Test_data["missing_t"][0, :]
    missing_t_test[1:] = missing_t_test[1:]-24

    wind_areas = ["DK1-1", "DK1-2", "DK1-3", "DK1-4", "DK1-5", "DK1-6", "DK1-7", "DK1-8", "DK1-9", "DK1-10", "DK1-11", "DK1-12", "DK1-13", "DK1-14", "DK1-15", "DK2-1", "DK2-2", "DK2-3", "DK2-4", "DK2-5", "DK2-6"]

    model = "model513_3_test"

    # =============================================================================
    # Hyperparameters
    # =============================================================================
    p = 12
    q = 0

    d = 1
    d_s = 0

    p_s = 0
    q_s = 0

    s = 288

    m = 0
    m_s = 0

    r_part = 13
    tau_ahead = 2

    model_name = model
    save_path = "VARIMA_Results/"+model_name

    # =============================================================================
    # Data Retrieval
    # =============================================================================
    if d == 1:
        Power_train = train_y
        y_train = Power_train[1:, :] - Power_train[:-1, :]
        n_train = np.shape(y_train)[0]
        missing_t_train[1:] = missing_t_train[1:]-1

        Power_test = test_y
        y_test = Power_test[1:, :] - Power_test[:-1, :]
        n_test = np.shape(y_test)[0]
        missing_t_test[1:] = missing_t_test[1:]-1

        Training_data = {"y" : y_train, "z_NWP": train_z_NWP,
                         "z_reg": train_z_reg, "missing_t" : missing_t_train}
        Test_data = {"y" : y_test, "z_NWP": test_z_NWP,
                     "z_reg": test_z_reg, "missing_t" : missing_t_test}
    elif d == 0:
        Power_train = train_y
        y_train = Power_train

        Power_test = test_y
        y_test = Power_test

        Training_data = {"y" : y_train, "z_NWP": train_z_NWP,
                         "z_reg": train_z_reg, "missing_t" : missing_t_train}
        Test_data = {"y" : y_test, "z_NWP": test_z_NWP,
                     "z_reg": test_z_reg, "missing_t" : missing_t_test}

    # =============================================================================
    # Model Fit and Validation
    # =============================================================================

    mod = sVARMAX_quick_fit(p, d, q, p_s, q_s, s, m, m_s, Training_data, r_part)
    mod.fit()
    Phi, Psi, Xi, Sigma_u = mod.return_parameters()

    P_max = np.max(Power_train, axis=0)

    if d == 1:
        MSE, NMAE, eps = mod.test_newer(tau_ahead, Test_data, P_max, 
                                        Power_test)
    elif d == 0:
        MSE, NMAE, eps = mod.test_newer(tau_ahead, Test_data, P_max)

    save_dict = {"Model": "s-VARIMAX({}, {}, {}) x ({}, {}, {})_{} with s-VARX({}) x ({})_{}".format(p, d, q, p_s, d_s, q_s, s, m, m_s, s),
                 "Xi": Xi, "Sigma_u": Sigma_u,
                 "MSE": MSE, "NMAE": NMAE}
    if p != 0 or q_s != 0:
        save_dict.update({"Phi": Phi})
    if q != 0 or q_s != 0:
        save_dict.update({"Psi": Psi})
    savemat(save_path+".mat", save_dict)