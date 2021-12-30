# -*- coding: utf-8 -*-
"""

Created on Mon Dec 13 11:37:13 2021

Authors:  Andreas Anton Andersen, Martin Voigt Vejling, and Morten Stig Kaaber
E-Mails: {aand17, mvejli17, mkaabe17}@student.aau.dk

This script can be used open and display the test results made with
    - Parallel_RNN.py

The script has been developed using Python 3.9 with the
library numpy.

"""

import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    model = "U_RNN_001" # Give input here

    wind_areas = ["DK1-1", "DK1-2", "DK1-3", "DK1-4", "DK1-5", "DK1-6", "DK1-7", "DK1-8", "DK1-9", "DK1-10", "DK1-11", "DK1-12", "DK1-13", "DK1-14", "DK1-15", "DK2-1", "DK2-2", "DK2-3", "DK2-4", "DK2-5", "DK2-6"]

    tau_ahead = 288
    total_MSE = np.zeros((tau_ahead, 21))
    total_NMAE = np.zeros((tau_ahead, 21))

    fontsize = 12
    plt.style.use("seaborn-darkgrid")
    params = {'axes.titlesize': fontsize,
              'axes.labelsize': fontsize,
              'xtick.labelsize': fontsize,
              'ytick.labelsize': fontsize}
    plt.rcParams.update(params)

    for l, area in enumerate(wind_areas):
        model_name = f"{model}_{area}"
        load_path_MSE = f"U-RNN/Test_MSE_{model_name}"
        load_path_NMAE = f"U-RNN/Test_NMAE_{model_name}"
        MSE = np.load(f"{load_path_MSE}.npy")
        NMAE = np.load(f"{load_path_NMAE}.npy")

        print(f"MSE {area}: {MSE[0]}")
        print(f"NMAE {area}: {NMAE[0]}")

        plt.plot(MSE)
        plt.ylabel("MSE")
        plt.xlabel(r"$\tau$")
        plt.title(area)
        plt.show()

        plt.plot(NMAE)
        plt.ylabel("NMAE")
        plt.xlabel(r"$\tau$")
        plt.title(area)
        plt.show()

        total_MSE[:, l] = MSE
        total_NMAE[:, l] = NMAE

    mean_MSE = np.mean(total_MSE, axis=1)
    mean_NMAE = np.mean(total_NMAE, axis=1)

    plt.plot(mean_MSE)
    plt.ylabel("Mean MSE")
    plt.xlabel(r"$\tau$")
    plt.show()

    plt.plot(mean_NMAE)
    plt.ylabel("Mean NMAE")
    plt.xlabel(r"$\tau$")
    plt.show()

