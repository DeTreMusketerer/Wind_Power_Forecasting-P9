# -*- coding: utf-8 -*-
"""

Created on Tue Jan  4 13:17:41 2022

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import scipy.stats as stats
from pandas import date_range
import matplotlib.dates as mdates



# fig, ax = plt.subplots(1, 1)
# ax.plot(index, a)
# plt.title('Wind Power Production in '+area+'[MW]')
# plt.ylabel('Wind Power Production [MW]')
# locator = mdates.AutoDateLocator(minticks=minticks, maxticks=maxticks)
# formatter = mdates.ConciseDateFormatter(locator)
# ax.xaxis.set_major_locator(locator)
# ax.xaxis.set_major_formatter(formatter)
# if savename is not None:
#     plt.savefig(savename, dpi=400)
# plt.show()

if __name__ == '__main__':
    model = "model001_test" # Give input here

    wind_areas = ["DK1-1", "DK1-2", "DK1-3", "DK1-4", "DK1-5", "DK1-6", "DK1-7", "DK1-8", "DK1-9", "DK1-10", "DK1-11", "DK1-12", "DK1-13", "DK1-14", "DK1-15", "DK2-1", "DK2-2", "DK2-3", "DK2-4", "DK2-5", "DK2-6"]

    tau_ahead = 288
    total_MSE = np.zeros((tau_ahead, 21))
    total_NMAE = np.zeros((tau_ahead, 21))

    area = wind_areas[0]
    model_name = f"{model}_{area}"
    load_path = f"Results/s-ARIMAX/{model_name}"
    mat = loadmat(load_path+".mat")
    eps = mat["eps"]
    _, n, _ = eps.shape
    total_eps = np.zeros(n)

    fontsize = 12
    plt.style.use("seaborn-darkgrid")
    params = {'axes.titlesize': fontsize,
              'axes.labelsize': fontsize,
              'xtick.labelsize': fontsize,
              'ytick.labelsize': fontsize}
    plt.rcParams.update(params)

    time = date_range("2020-10-05 12:00:00", "2021-10-05 09:55:00", freq="5min")

    for l, area in enumerate(wind_areas):
        model_name = f"{model}_{area}"
        load_path = f"Results/s-ARIMAX/{model_name}"
        mat = loadmat(load_path+".mat")
        eps = mat["eps"]
        total_eps += eps[0, :, 0]

        # fig, ax = plt.subplots(1, 1)
        # ax.plot(time, eps[0, :, 0])
        # plt.ylabel("Residuals")
        # plt.title(f"{area}")
        # locator = mdates.AutoDateLocator(minticks=3, maxticks=10)
        # formatter = mdates.ConciseDateFormatter(locator)
        # ax.xaxis.set_major_locator(locator)
        # ax.xaxis.set_major_formatter(formatter)
        # plt.show()

        # plt.hist(eps[0, :, 0], bins=100)
        # plt.title("Histogram")
        # plt.show()

        # stats.probplot(eps[0, :, 0], dist="norm", plot=plt)
        # plt.show()


    fig, ax = plt.subplots(1, 1)
    ax.plot(time, total_eps)
    plt.ylabel("Residuals")
    plt.title("Average")
    locator = mdates.AutoDateLocator(minticks=3, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.savefig("residual_plot.png", dpi=400)
    plt.show()

    plt.hist(total_eps, bins=100)
    plt.title("Histogram")
    plt.show()

    stats.probplot(total_eps, dist="norm", plot=plt)
    plt.savefig("residual_qq.png", dpi=400)
    plt.show()
