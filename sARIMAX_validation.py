# -*- coding: utf-8 -*-
"""

Created on Fri Nov 25 12:43:54 2021

Authors:  Andreas Anton Andersen, Martin Voigt Vejling, and Morten Stig Kaaber
E-Mails: {aand17, mvejli17, mkaabe17}@student.aau.dk

This script can be used to train and validate s-ARIMAX models, see the report
        Forecasting Wind Power Production
            - Chapter 2: Time Series Analysis
            - Chapter 6: Experimental Setup
                - Section 6.2.2: s-ARIMAX

The script has been developed using Python 3.9 with the
libraries numpy and scipy.

"""

import numpy as np
from scipy.io import loadmat, savemat

from Modules.sVARMAX_Module import sVARMAX


if __name__ == '__main__':
    Training_data = loadmat("data_energinet/New_Training_Data.mat")
    Power_train = Training_data["y"]

    Subtraining_data = loadmat("data_energinet/New_Subtraining_Data.mat")
    subtrain_y = Subtraining_data["y"]
    subtrain_z_NWP = Subtraining_data["z_NWP"]
    subtrain_z_reg = Subtraining_data["z_reg"]
    missing_t_subtrain = Subtraining_data["missing_t"][0, :]

    Validation_data = loadmat("data_energinet/New_Validation_Data.mat")
    validation_y = Validation_data["y"]
    validation_z_NWP = Validation_data["z_NWP"]
    validation_z_reg = Validation_data["z_reg"]
    missing_t_validation = Validation_data["missing_t"][0, :]
    #missing_t_validation[-1] = missing_t_validation[-1] - 1

    wind_areas = ["DK1-1", "DK1-2", "DK1-3", "DK1-4", "DK1-5", "DK1-6", "DK1-7", "DK1-8", "DK1-9", "DK1-10", "DK1-11", "DK1-12", "DK1-13", "DK1-14", "DK1-15", "DK2-1", "DK2-2", "DK2-3", "DK2-4", "DK2-5", "DK2-6"]

    model = "model001_validation"

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
    tau_ahead = 1

    for l, area in enumerate(wind_areas):
        model_name = f"{model}_{area}"
        save_path = f"Results/s-ARIMAX/{model_name}"

        Power_subtrain = np.expand_dims(subtrain_y[:, l], -1)

        mod = sVARMAX(Power_subtrain, subtrain_z_reg, subtrain_z_NWP, np.copy(missing_t_subtrain),
                      p, d, q, p_s, q_s, s, m, m_s, l)
        mod.fit()
        Phi, Psi, Xi, Sigma_u = mod.return_parameters()

        P_max = np.max(Power_train[:, l])
        Power_validation = np.expand_dims(validation_y[:, l], -1)

        MSE, NMAE, eps = mod.test(tau_ahead, Power_validation, validation_z_reg, validation_z_NWP,
                                  np.copy(missing_t_validation), P_max)

        save_dict = {"Wind area": area+" Validation",
                      "Model": "s-ARIMAX({}, {}, {}) x ({}, {}, {})_{} with s-ARX({}) x ({})_{}".format(p, d, q, p_s, d_s, q_s, s, m, m_s, s),
                      "Xi": Xi, "Sigma_u": Sigma_u,
                      "MSE": MSE[:, 0], "NMAE": NMAE[:, 0]}
        if p != 0 or q_s != 0:
            save_dict.update({"Phi": Phi})
        if q != 0 or q_s != 0:
            save_dict.update({"Psi": Psi})
        savemat(save_path+".mat", save_dict)