# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 08:38:30 2021

@author: marti
"""


import os
import inspect

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat


if __name__ == '__main__':
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)

    np.random.seed(49)

    wind_areas = ["DK1-1", "DK1-2", "DK1-3", "DK1-4", "DK1-5", "DK1-6", "DK1-7", "DK1-8", "DK1-9", "DK1-10", "DK1-11", "DK1-12", "DK1-13", "DK1-14", "DK1-15", "DK2-1", "DK2-2", "DK2-3", "DK2-4", "DK2-5", "DK2-6"]

    total_MSE = np.zeros((2, 21))
    total_NMAE = np.zeros((2, 21))
    model = "model310_3"

    fontsize = 12
    plt.style.use("seaborn-darkgrid")
    params = {'axes.titlesize': fontsize,
              'axes.labelsize': fontsize,
              'xtick.labelsize': fontsize,
              'ytick.labelsize': fontsize}
    plt.rcParams.update(params)

    for l, area in enumerate(wind_areas):
        model_name = model+"_"+area
        save_path = "ARIMA_Results/"+model+"/"+model_name
        mat = loadmat(save_path+".mat")
        MSE = mat["MSE"]
        NMAE = mat["NMAE"]
        print(f"MSE {area}: {MSE[0, 0]}")
        print(f"NMAE {area}: {NMAE[0, 0]}")

        plt.plot(MSE[0, :])
        plt.ylabel("MSE")
        plt.xlabel(r"$\tau$")
        plt.title(area)
        #plt.savefig(save_path+"_MSE"+".png", dpi=400)
        plt.show()

        plt.plot(NMAE[0, :])
        plt.ylabel("NMAE")
        plt.xlabel(r"$\tau$")
        plt.title(area)
        #plt.savefig(save_path+"_NMAE"+".png", dpi=400)
        plt.show()

        total_MSE[:, l] = MSE
        total_NMAE[:, l] = NMAE

        # save_dict = {"Wind area": area+" Validation",
        #               "Model": mat["Model"],
        #               "Xi": mat["Xi"], "Sigma_u": mat["Sigma_u"],
        #               "MSE": mat["MSE"][:, 0], "NMAE": mat["NMAE"][:, l]}
        # try:
        #     save_dict.update({"Phi": mat["Phi"]})
        # except:
        #     pass
        # try:
        #     save_dict.update({"Psi": mat["Psi"]})
        # except:
        #     pass
        # savemat(save_path+".mat", save_dict)


    mean_MSE = np.mean(total_MSE, axis=1)
    mean_NMAE = np.mean(total_NMAE, axis=1)

    plt.plot(mean_MSE)
    plt.ylabel("Mean MSE")
    plt.xlabel(r"$\tau$")
    #plt.savefig("ARIMA_Results/"+model+"/"+model+"_Mean_MSE"+".png", dpi=400)
    plt.show()

    plt.plot(mean_NMAE)
    plt.ylabel("Mean NMAE")
    plt.xlabel(r"$\tau$")
    #plt.savefig("ARIMA_Results/"+model+"/"+model+"_Mean_NMAE"+".png", dpi=400)
    plt.show()



