# -*- coding: utf-8 -*-
"""

Created on Fri Nov 26 10:02:12 2021

Authors:  Andreas Anton Andersen, Martin Voigt Vejling, and Morten Stig Kaaber
E-Mails: {aand17, mvejli17, mkaabe17}@student.aau.dk

This script can be used to train and test s-VARIMAX models, see the report
        Forecasting Wind Power Production
            - Chapter 2: Time Series Analysis
            - Chapter 6: Experimental Setup
                - Section 6.2.4: s-VARIMAX

The script has been developed using Python 3.9 with the
libraries numpy and scipy.

"""

import numpy as np
from scipy.io import loadmat, savemat

from Modules.sVARMAX_Module import sVARMAX


if __name__ == '__main__':
    Training_data = loadmat("data_energinet/New_Training_Data.mat")

    train_y = Training_data["y"]
    train_z_NWP = Training_data["z_NWP"]
    train_z_reg = Training_data["z_reg"]
    missing_t_train = Training_data["missing_t"][0, :]

    Test_data = loadmat("data_energinet/New_Test_Data.mat")
    test_y = Test_data["y"][24:, :] # Need to start at 12:00:00 to match NWP
    test_z_NWP = Test_data["z_NWP"]
    test_z_reg = Test_data["z_reg"][24:, :]
    missing_t_test = Test_data["missing_t"][0, :]
    missing_t_test[1:] = missing_t_test[1:]-24

    wind_areas = ["DK1-1", "DK1-2", "DK1-3", "DK1-4", "DK1-5", "DK1-6", "DK1-7", "DK1-8", "DK1-9", "DK1-10", "DK1-11", "DK1-12", "DK1-13", "DK1-14", "DK1-15", "DK2-1", "DK2-2", "DK2-3", "DK2-4", "DK2-5", "DK2-6"]

    model = "model001_test"

    # =============================================================================
    # Hyperparameters
    # =============================================================================
    p = 1
    q = 0

    d = 1
    d_s = 0

    p_s = 0
    q_s = 0

    s = 288

    m = 0
    m_s = 0

    r_part = 13
    tau_ahead = 1

    model_name = model
    save_path = f"Results/s-VARIMAX/{model_name}"


    # =============================================================================
    # Model Fit and Validation
    # =============================================================================

    mod = sVARMAX(train_y, train_z_reg, train_z_NWP, np.copy(missing_t_train),
                  p, d, q, p_s, q_s, s, m, m_s)
    mod.fit()
    Phi, Psi, Xi, Sigma_u = mod.return_parameters()

    P_max = np.max(test_y, axis=0)
    Power_test = test_y

    MSE, NMAE, eps = mod.test(tau_ahead, test_y, test_z_reg, test_z_NWP,
                              np.copy(missing_t_test), P_max, Power_test)

    save_dict = {"Model": "s-VARIMAX({}, {}, {}) x ({}, {}, {})_{} with s-VARX({}) x ({})_{}".format(p, d, q, p_s, d_s, q_s, s, m, m_s, s),
                 "Xi": Xi, "Sigma_u": Sigma_u,
                 "MSE": MSE, "NMAE": NMAE}
    if p != 0 or q_s != 0:
        save_dict.update({"Phi": Phi})
    if q != 0 or q_s != 0:
        save_dict.update({"Psi": Psi})
    savemat(save_path+".mat", save_dict)