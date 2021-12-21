# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 09:54:24 2021

Authors:  Andreas Anton Andersen, Martin Voigt Vejling, and Morten Stig Kaaber
E-Mails: {aand17, mvejli17, mkaabe17}@student.aau.dk

This script can be used open and display the test results made with
    - sVARIMAX_test.py
    - sVARIMAX_validation.py

The script has been developed using Python 3.9 with the
libraries numpy and scipy.

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


if __name__ == '__main__':
    model = "model001_validation" # Give input here


    wind_areas = ["DK1-1", "DK1-2", "DK1-3", "DK1-4", "DK1-5", "DK1-6", "DK1-7", "DK1-8", "DK1-9", "DK1-10", "DK1-11", "DK1-12", "DK1-13", "DK1-14", "DK1-15", "DK2-1", "DK2-2", "DK2-3", "DK2-4", "DK2-5", "DK2-6"]

    fontsize = 12
    plt.style.use("seaborn-darkgrid")
    params = {'axes.titlesize': fontsize,
              'axes.labelsize': fontsize,
              'xtick.labelsize': fontsize,
              'ytick.labelsize': fontsize}
    plt.rcParams.update(params)

    model_name = model
    load_path = f"s-VARIMAX/{model_name}"
    mat = loadmat(load_path+".mat")
    MSE = mat["MSE"]
    NMAE = mat["NMAE"]

    for l, area in enumerate(wind_areas):
        plt.plot(MSE[:, l])
        plt.ylabel("MSE")
        plt.xlabel(r"$\tau$")
        plt.title(area)
        plt.show()
        print(f"MSE {area}: {MSE[0, l]}")
        print(f"NMAE {area}: {NMAE[0, l]}")

        plt.plot(NMAE[:, l])
        plt.ylabel("NMAE")
        plt.xlabel(r"$\tau$")
        plt.title(area)
        plt.show()

    mean_MSE = np.mean(MSE, axis=1)
    mean_NMAE = np.mean(NMAE, axis=1)

    plt.plot(mean_MSE)
    plt.ylabel("Mean MSE")
    plt.xlabel(r"$\tau$")
    plt.show()

    plt.plot(mean_NMAE)
    plt.ylabel("Mean NMAE")
    plt.xlabel(r"$\tau$")
    plt.show()



