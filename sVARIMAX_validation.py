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
    Power_train = Training_data["y"]

    Subtraining_data = loadmat(parentdir+"/data_energinet/New_Subtraining_Data.mat")
    subtrain_y = Subtraining_data["y"]
    subtrain_z_NWP = Subtraining_data["z_NWP"]
    subtrain_z_reg = Subtraining_data["z_reg"]
    missing_t_subtrain = Subtraining_data["missing_t"][0, :]

    Validation_data = loadmat(parentdir+"/data_energinet/New_Validation_Data.mat")
    validation_y = Validation_data["y"]
    validation_z_NWP = Validation_data["z_NWP"]
    validation_z_reg = Validation_data["z_reg"]
    missing_t_validation = Validation_data["missing_t"][0, :]

    wind_areas = ["DK1-1", "DK1-2", "DK1-3", "DK1-4", "DK1-5", "DK1-6", "DK1-7", "DK1-8", "DK1-9", "DK1-10", "DK1-11", "DK1-12", "DK1-13", "DK1-14", "DK1-15", "DK2-1", "DK2-2", "DK2-3", "DK2-4", "DK2-5", "DK2-6"]

    model = "model520_3"

    # =============================================================================
    # Hyperparameters
    # =============================================================================
    p = 36
    q = 0

    d = 1
    d_s = 0

    p_s = 2
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
        Power_subtrain = subtrain_y
        y_subtrain = Power_subtrain[1:, :] - Power_subtrain[:-1, :]
        n_subtrain = np.shape(y_subtrain)[0]
        missing_t_subtrain[1:] = missing_t_subtrain[1:]-1

        Power_validation = validation_y
        y_validation = Power_validation[1:, :] - Power_validation[:-1, :]
        n_validation = np.shape(y_validation)[0]
        missing_t_validation[1:] = missing_t_validation[1:]-1

        Training_data = {"y" : y_subtrain, "z_NWP": subtrain_z_NWP,
                         "z_reg": subtrain_z_reg, "missing_t" : missing_t_subtrain}
        Validation_data = {"y" : y_validation, "z_NWP": validation_z_NWP,
                           "z_reg": validation_z_reg, "missing_t" : missing_t_validation}
    elif d == 0:
        Power_subtrain = subtrain_y
        y_subtrain = Power_subtrain

        Power_validation = validation_y
        y_validation = Power_validation

        Training_data = {"y" : y_subtrain, "z_NWP": subtrain_z_NWP,
                         "z_reg": subtrain_z_reg, "missing_t" : missing_t_subtrain}
        Validation_data = {"y" : y_validation, "z_NWP": validation_z_NWP,
                           "z_reg": validation_z_reg, "missing_t" : missing_t_validation}

    # =============================================================================
    # Model Fit and Validation
    # =============================================================================

    mod = sVARMAX_quick_fit(p, d, q, p_s, q_s, s, m, m_s, Training_data, r_part)
    mod.fit()
    Phi, Psi, Xi, Sigma_u = mod.return_parameters()

    P_max = np.max(Power_train, axis=0)

    if d == 1:
        MSE, NMAE, eps = mod.test_newer(tau_ahead, Validation_data, P_max, 
                                        Power_validation)
    elif d == 0:
        MSE, NMAE, eps = mod.test_newer(tau_ahead, Validation_data, P_max)

    save_dict = {"Model": "s-VARIMAX({}, {}, {}) x ({}, {}, {})_{} with s-VARX({}) x ({})_{}".format(p, d, q, p_s, d_s, q_s, s, m, m_s, s),
                 "Xi": Xi, "Sigma_u": Sigma_u,
                 "MSE": MSE, "NMAE": NMAE}
    if p != 0 or q_s != 0:
        save_dict.update({"Phi": Phi})
    if q != 0 or q_s != 0:
        save_dict.update({"Psi": Psi})
    savemat(save_path+".mat", save_dict)