# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 11:23:03 2021

@author: marti
"""


import os
import inspect

import numpy as np
from scipy.io import loadmat, savemat
from time import time

from sVARMAX.sVARMAX_Core import sVARMAX_model_core

#import line_profiler
#profile = line_profiler.LineProfiler()


class sVARMAX_quick_fit(sVARMAX_model_core):
    """
    Compute parameters of a s-VARMAX model using OLS.
    """
    def __init__(self, p, d, q, p_s, q_s, s, m, m_s, data, r_part, l=0, other_z=False, EMD=False, reg=True):
        """
        Parameters
        ----------
        p : int
            Autoregressive order.
        q : int
            Moving average order.
        d : int
            Order of differencing.
        p_s : int, optional
            Seasonal autoregressive order.
        q_s : int, optional
            Seasonal moving average order.
        s : int, optional
            Seasonal delay.
        m : int
            Order of autoregressive model used for the initial parameter
            estimate.
        m_s : int
            Seasonal order for the autoregressive model used for the initial
            parameter estimate.
        data : dict
            z_reg : ndarray, size=(n, 2)
                Regulation data.
            z_NWP : ndarray, size=(55, n_nwp, 11*k)
                Numerical weather prediction data.
            y : ndarray, size=(n, k)
                Power data.
            missing_t : ndarray
                Array of time indices where a discontinuity in time is present due
                to missing power history data. The first entry in the list is
                zero and the last entry in the list is n.
        r_data : int
            Number of exogenous variables per wind area.
        l : int, optional
            Wind area. The default is 0.
        other_z : bool, optional
            Whether or not to use my other z. The default is False.
        EMD : bool, optional
            If EMD, then don't use NWP data. The default is False.
        """
        # Store data
        #self.store_data_pairs(data)
        self.y = data["y"].astype(dtype=np.float32)
        self.z_reg = data["z_reg"].astype(dtype=np.float32)
        self.z_NWP = data["z_NWP"].astype(dtype=np.float32)

        # Initialize variables
        self.n, self.k = np.shape(self.y)
        self.r_part = r_part
        self.r = self.k*self.r_part

        self.p = p
        self.q = q
        self.p_s = p_s
        self.q_s = q_s
        self.s = s
        self.d = d
        self.l = l
        self.m = m
        self.m_s = m_s

        self.max_delay_AR = p_s*s + p # The maximum delay in the autoregressive part
        self.max_delay_MA = q_s*s + q # The maximum delay in the moving average part
        self.max_delay = max(self.max_delay_AR, self.max_delay_MA) # The maximum delay
        self.p_tot = p+(p+1)*p_s # Total number of autoregressive delays
        self.q_tot = q+(q+1)*q_s # Total number of moving average delays
        if self.max_delay_AR >= self.max_delay_MA:
            self.xdim = self.max_delay_AR*self.k
        elif self.max_delay_AR < self.max_delay_MA:
            self.xdim = self.max_delay_MA*self.k

        # Assertions
        assert self.r_part*self.k == self.r or self.r_part == self.r
        if p_s == 0: assert m_s == 0
        if EMD is True and reg is True:
            assert self.r == 2
            assert other_z is False
        elif EMD is True and reg is False:
            assert self.r == 1
            assert other_z is False            

        # Missing data time indices
        self.missing_t = data["missing_t"]
        self.nr_missing_t = len(self.missing_t)-1

        if EMD is True:
            self.make_z = self.make_EMD_z
        elif other_z is True:
            self.make_z = self.make_other_z
        else:
            self.make_z = self.make_NWP_z

    def fit(self, reg=True):
        self.z = np.zeros((self.n, self.r))
        for t in range(self.n):
            if reg is True:
                self.z[t, :] = self.make_z(0, t, self.z_reg, self.z_NWP).astype(dtype=np.float32)
            elif reg is False:
                self.z[t, :] = self.make_z(0, t, self.z_reg, self.z_NWP, reg).astype(dtype=np.float32)

        if self.q == 0:
            Xi, Phi, Sigma_u, _ = self.sVARX_fit(self.p, self.p_s, self.s)
            Theta = {"Phi": Phi, "Psi": None, "Xi": Xi, "Sigma_u": Sigma_u}
        else:
            # Do initial parameter estimation
            Xi, Phi, Psi, Sigma_u = self.sVARMAX_fit(self.m, self.m_s)
            Theta = {"Phi": Phi, "Psi": Psi, "Xi": Xi, "Sigma_u": Sigma_u}

        self.update_parameters(Theta)

    def sVARX_fit(self, p, p_s, s):
        """
        Fit a s-VARX(p) x (p_s)_s to x_hat using OLS.
        """

        print("Fit a s-VARX({}) x ({})_{} model.".format(p, p_s, s))

        delay_list_AR = [j_s*s+j for j_s in range(p_s+1) for j in range(p+1)][1:]
        max_delay_AR = p_s*s + p
        p_tot = p+(p+1)*p_s

        u_hat_temp = np.zeros((self.n-2*288-(max_delay_AR*self.nr_missing_t), self.k), dtype=np.float32)
        pars = np.zeros((p_tot*self.k+self.r_part, self.k), dtype=np.float32)

        idx_list = []

        for l in range(self.l, self.l+self.k):
            idx = 0
            Y = np.zeros(self.n-2*288-(max_delay_AR*self.nr_missing_t), dtype=np.float32)
            X = np.zeros((self.n-2*288-(max_delay_AR*self.nr_missing_t), p_tot*self.k+self.r_part), dtype=np.float32)
            for missing_t_idx in range(self.nr_missing_t):
                idx_list.append(idx)
                a = self.missing_t[missing_t_idx]+max_delay_AR
                if missing_t_idx < self.nr_missing_t-1:
                    b = self.missing_t[missing_t_idx+1]-288
                else:
                    b = self.missing_t[missing_t_idx+1]
                for t in range(a, b):
                    X_t = np.zeros((p_tot, self.k))
                    for counter, delay in enumerate(delay_list_AR):
                        X_t[counter, :] = self.y[t-delay, :]
                    X[idx, :p_tot*self.k] = X_t.flatten()
                    if self.k == 1:
                        X[idx, p_tot*self.k:] = self.z[t, :]
                        Y[idx] = self.y[t, 0]
                    elif self.k == 21:
                        X[idx, p_tot*self.k:] = self.z[t, l*self.r_part:(l+1)*self.r_part]
                        Y[idx] = self.y[t, l]
                    idx += 1
            idx_list.append(idx)
            if self.k == 1:
                pars[:, 0], u_hat_temp[:, 0] = self.multivariate_OLS(Y, X)
            elif self.k == 21:
                pars[:, l], u_hat_temp[:, l] = self.multivariate_OLS(Y, X)
        zeros = np.zeros((max_delay_AR+288, self.k), dtype=np.float32)
        u_hat = np.concatenate((np.zeros((max_delay_AR, self.k)), u_hat_temp[idx_list[0]:idx_list[1], :]), axis=0)
        u_hat = np.concatenate((u_hat, zeros, u_hat_temp[idx_list[1]:idx_list[2], :]), axis=0)
        u_hat = np.concatenate((u_hat, zeros, u_hat_temp[idx_list[2]:idx_list[3], :]), axis=0)

        Phi = [pars[j*self.k:(j+1)*self.k, :] for j in range(p_tot)]
        Xi = np.zeros((self.k, self.r), dtype=np.float32)
        if self.k == 1:
            Xi[0, :] = pars[p_tot*self.k:, 0]
        elif self.k == 21:
            for l in range(self.k):
                Xi[l, l*self.r_part:(l+1)*self.r_part] = pars[p_tot*self.k:, l]
        Sigma_u = np.sum(np.array([np.outer(u_hat[t, :], u_hat[t, :]) for t in range(self.n-2*288-(max_delay_AR*self.nr_missing_t))]), axis=0)/(self.n-2*288-(max_delay_AR*self.nr_missing_t)-1)
        return Xi, Phi, Sigma_u, u_hat

    def sVARMAX_fit(self, m, m_s):
        """
        Fit s-VARMAX using OLS.

        1) Fit a s-VARX(m) x (m_s)_s for m >> p and m_s >> p_s model to y
           using OLS. Compute the residuals u_hat for the resulting model.
        2) Using u_hat do OLS to estimate the s-VARMAX(p, q) x (p_s, q_s)_s
           parameters.
        """

        if self.p_s != 0: assert self.s > m

        _, _, _, u_hat = self.sVARX_fit(m, m_s, self.s)

        print("Fit a s-VARMAX({}, {}) x ({}, {})_{} model.".format(
            self.p, self.q, self.p_s, self.q_s, self.s))

        # Step 2)
        delay_list_AR = [j_s*self.s+j for j_s in range(self.p_s+1) for j in range(self.p+1)][1:]
        delay_list_MA = [i_s*self.s+i for i_s in range(self.q_s+1) for i in range(self.q+1)][1:]

        pars = np.zeros(((self.p_tot+self.q_tot)*self.k + self.r_part, self.k), dtype=np.float32)
        u_hat_new = np.zeros((self.n-2*288-(self.max_delay_AR*self.nr_missing_t), self.k), dtype=np.float32)

        for l in range(self.l, self.l+self.k):
            idx = 0
            Y = np.zeros(self.n-2*288-(self.max_delay_AR*self.nr_missing_t), dtype=np.float32)
            X = np.zeros((self.n-2*288-(self.max_delay_AR*self.nr_missing_t), (self.p_tot+self.q_tot)*self.k + self.r_part), dtype=np.float32)
            for missing_t_idx in range(self.nr_missing_t):
                a = self.missing_t[missing_t_idx]+self.max_delay_AR
                if missing_t_idx < self.nr_missing_t-1:
                    b = self.missing_t[missing_t_idx+1]-288
                else:
                    b = self.missing_t[missing_t_idx+1]
                for t in range(a, b):
                    X_t_AR = np.zeros((self.p_tot, self.k), dtype=np.float32)
                    X_t_MA = np.zeros((self.q_tot, self.k), dtype=np.float32)
                    for counter, delay_AR in enumerate(delay_list_AR):
                        X_t_AR[counter, :] = self.y[t-delay_AR, :]
                    for counter, delay_MA in enumerate(delay_list_MA):
                        X_t_MA[counter, :] = -u_hat[t-delay_MA, :]
                    X[idx, :(self.p_tot+self.q_tot)*self.k] = np.vstack((X_t_AR, X_t_MA)).flatten()
                    if self.k == 1:
                        X[idx, (self.p_tot+self.q_tot)*self.k:] = self.z[t, :]
                        Y[idx] = self.y[t, 0]
                    elif self.k == 21:
                        X[idx, (self.p_tot+self.q_tot)*self.k:] = self.z[t, l*self.r_part:(l+1)*self.r_part]
                        Y[idx] = self.y[t, l]
                    idx += 1
            if self.k == 1:
                pars[:, 0], u_hat_new[:, 0] = self.multivariate_OLS(Y, X)
            elif self.k == 21:
                pars[:, l], u_hat_new[:, l] = self.multivariate_OLS(Y, X)
        Phi = [pars[j*self.k:(j+1)*self.k, :] for j in range(self.p_tot)]
        Psi = [pars[self.p_tot*self.k+i*self.k:self.p_tot*self.k+(i+1)*self.k, :] for i in range(self.q_tot)]
        Xi = np.zeros((self.k, self.r), dtype=np.float32)
        if self.k == 1:
            Xi[0, :] = pars[(self.p_tot+self.q_tot)*self.k:, 0]
        elif self.k == 21:
            for l in range(self.k):
                Xi[l, l*self.r_part:(l+1)*self.r_part] = pars[(self.p_tot+self.q_tot)*self.k:, l]
        Sigma_u = np.sum(np.array([np.outer(u_hat_new[t, :], u_hat_new[t, :]) for t in range(self.n-2*288-(self.max_delay_AR*self.nr_missing_t))]), axis=0)/(self.n-2*288-(self.max_delay_AR*self.nr_missing_t)-1)
        return Xi, Phi, Psi, Sigma_u

    def multivariate_OLS(self, Y, X):
        t1 = time()
        B = np.linalg.inv(X.T @ X) @ X.T @ Y
        t2 = time()
        print("Parameter fit time: {}".format(t2-t1))
        eps = Y - X @ B
        return B, eps

    # def tau_ahead_forecast(self, tau_ahead, t, test_data, P_test = None):
    #     """
    #     Compute the tau-ahead forecast using the truncated forecasting as
    #     defined in property 3.7 of Shumway2017.

    #     Parameters
    #     ----------
    #     tau_ahead : int
    #         Compute tau-ahead forecast given time t-1.
    #     t : int
    #         Time index in the test set.
    #     test_data : dict
    #         y_test : ndarray, size=(n_test, k)
    #             Endogenous variable.
    #         z_NWP_test : ndarray, size=(tau_ahead, nwp_n_test, 11*k)
    #             Numerical weather predictions with the given transformations.
    #             The first axis is the tau_ahead axis while the second axis gives
    #             a new set of NWP. The last axis is as follows:
    #                 (T, sin(WD10m), cos(WD10m), sin(WD100m), cos(WD100m),
    #                  WS10m, WS10m^2, WS10m^3, WS100m, WS100m^2, WS100m^3).
    #         z_reg_test : ndarray, size=(n_test, 2)
    #             Regulation data for DK1 in the first column and DK2 in the second
    #             column.
    #         test_missing_t : list
    #             List of time indices where a discontinuity in time is present due
    #             to missing power history data. The first entry in the list is
    #             zero and the last entry in the list is n.
    #     P_test : ndarray, size=(k, ), optional
    #         Wind power at time t-2. Used when first order differencing is used.
    #         The default is None.

    #     Returns
    #     -------
    #     P_bar : ndarray, size=(tau_ahead, k)
    #         Wind power forecast.
    #     """

    #     y_test = test_data["y"]
    #     z_reg_test = test_data["z_reg"]
    #     z_NWP_test = test_data["z_NWP"]
    #     test_missing_t = test_data["missing_t"][0, :]

    #     array_AR = np.array([j_s*self.s+j for j_s in range(self.p_s+1) for j in range(self.p+1)])[1:]
    #     array_MA = np.array([i_s*self.s+i for i_s in range(self.q_s+1) for i in range(self.q+1)])[1:]

    #     test_nr_missing_t = len(test_missing_t)-1
    #     n_test, _ = y_test.shape

    #     assert t >= self.max_delay
    #     assert t < n_test - tau_ahead
    #     if test_nr_missing_t == 2:
    #         assert t >= test_missing_t[1] + self.max_delay or t < test_missing_t[1]-tau_ahead-288
    #     else:
    #         assert False

    #     y_bar = np.zeros((tau_ahead, self.k))
    #     if self.d == 1:
    #         P_bar = np.zeros((tau_ahead, self.k))

    #     idx_list = []

    #     u_hat = np.zeros((n_test, self.k))
    #     if t < test_missing_t[1]:
    #         for t_i in range(self.max_delay, t+tau_ahead):
    #             u_hat[t_i, :] += y_test[t_i, :]
    #             for j, idx in enumerate(array_AR):
    #                 u_hat[t_i, :] -= np.dot(self.Phi[j], y_test[t_i-idx, :])
    #             for i, idx in enumerate(array_MA):
    #                 u_hat[t_i, :] += np.dot(self.Psi[i], u_hat[t_i-idx, :])
    #             z_data = self.make_z(0, t_i, z_reg_test, z_NWP_test)
    #             u_hat[t_i, :] -= np.dot(self.Xi, z_data)
    #     elif t > test_missing_t[1]:
    #         for t_i in range(test_missing_t[1]+self.max_delay, t+tau_ahead):
    #             u_hat[t_i, :] += y_test[t_i, :]
    #             for j, idx in enumerate(array_AR):
    #                 u_hat[t_i, :] -= np.dot(self.Phi[j], y_test[t_i-idx, :])
    #             for i, idx in enumerate(array_MA):
    #                 u_hat[t_i, :] += np.dot(self.Psi[i], u_hat[t_i-idx, :])
    #             z_data = self.make_z(0, t_i, z_reg_test, z_NWP_test)
    #             u_hat[t_i, :] -= np.dot(self.Xi, z_data)

    #     # u_hat = self.estimate_noise_newer(tau_ahead, y_test, z_NWP_test,
    #     #                                   z_reg_test, test_missing_t)

    #     phi_mat = np.hstack(self.Phi)
    #     if self.q != 0 or self.q_s != 0:
    #         psi_mat = np.hstack(self.Psi)
    #         beta = np.concatenate((phi_mat, -psi_mat, self.Xi), axis=1)
    #     else:
    #         beta = np.concatenate((phi_mat, self.Xi), axis=1)

    #     # Compute the tau-ahead prediction for each time in the test set
    #     for tau_i in range(tau_ahead):
    #         if tau_i == 0:
    #             idx_list.append(t)
    #             y_vec = y_test[t-array_AR, :].flatten()
    #             u_vec = u_hat[t-array_MA, :].flatten()
    #             z_data = self.make_z(0, t, z_reg_test, z_NWP_test)
    #             data_vec = np.hstack((y_vec, u_vec, z_data))
    #             y_bar[0, :] = np.dot(beta, data_vec)
    #         else:
    #             bar_AR = array_AR[tau_i-array_AR >= 0]
    #             test_AR = array_AR[tau_i-array_AR < 0]
    #             hat_MA = array_MA[tau_i-array_MA < 0]
    #             if len(bar_AR) != 0:
    #                 y_vec_bar = y_bar[tau_i-bar_AR, :].flatten()
    #             else:
    #                 y_vec_bar = np.array([])
    #             if len(test_AR) != 0:
    #                 y_vec_test = y_test[t+tau_i-test_AR, :].flatten()
    #             else:
    #                 y_vec_test = np.array([])
    #             if len(hat_MA) != 0:
    #                 u_vec = u_hat[t+tau_i-hat_MA, :].flatten()
    #             else:
    #                 u_vec = np.array([])
    #             y_bar[tau_i, :] += np.dot(phi_mat, np.hstack((y_vec_bar, y_vec_test)))
    #             if self.q != 0 or self.q_s != 0:
    #                 y_bar[tau_i, :] -= np.dot(psi_mat[:, (len(array_MA)-len(hat_MA))*self.k:], u_vec)
    #             z_data = self.make_z(tau_i, t, z_reg_test, z_NWP_test)
    #             y_bar[tau_i, :] += np.dot(self.Xi, z_data)

    #         if self.d == 1:
    #             for tau_i in range(tau_ahead):
    #                 if tau_i == 0:
    #                     P_bar[tau_i, :] = y_bar[tau_i, :] + P_test
    #                 else:
    #                     P_bar[tau_i, :] = y_bar[tau_i, :] + P_bar[tau_i-1, :]
    #         else:
    #             P_bar = y_bar

    #     return P_bar

    def forecast(self, tau_ahead, t_start, t_end, test_data, P_test = None):
        """
        Compute the one-ahead forecast using the truncated forecasting as
        defined in property 3.7 of Shumway2017.

        Parameters
        ----------
        tau_ahead : int
            Compute tau-ahead forecast given time t-1.
        t_start : int
            Time index in the test set.
        t_end : int
            Time index in the test set to end.
        test_data : dict
            y_test : ndarray, size=(n_test, k)
                Endogenous variable.
            z_NWP_test : ndarray, size=(tau_ahead, nwp_n_test, 11*k)
                Numerical weather predictions with the given transformations.
                The first axis is the tau_ahead axis while the second axis gives
                a new set of NWP. The last axis is as follows:
                    (T, sin(WD10m), cos(WD10m), sin(WD100m), cos(WD100m),
                     WS10m, WS10m^2, WS10m^3, WS100m, WS100m^2, WS100m^3).
            z_reg_test : ndarray, size=(n_test, 2)
                Regulation data for DK1 in the first column and DK2 in the second
                column.
            test_missing_t : list
                List of time indices where a discontinuity in time is present due
                to missing power history data. The first entry in the list is
                zero and the last entry in the list is n.
        P_test : ndarray, size=(n_test+1, k), optional
            Wind power at time t-2. Used when first order differencing is used.
            The default is None.

        Returns
        -------
        P_bar : ndarray, size=(t_end-t_start, k)
            Wind power forecast.
        """

        t_length = t_end-t_start
        y_test = test_data["y"]
        z_reg_test = test_data["z_reg"]
        z_NWP_test = test_data["z_NWP"]
        test_missing_t = test_data["missing_t"]

        array_AR = np.array([j_s*self.s+j for j_s in range(self.p_s+1) for j in range(self.p+1)])[1:]
        array_MA = np.array([i_s*self.s+i for i_s in range(self.q_s+1) for i in range(self.q+1)])[1:]

        test_nr_missing_t = len(test_missing_t)-1
        n_test, _ = y_test.shape

        assert t_start >= self.max_delay
        assert t_end < n_test
        if test_nr_missing_t == 2:
            assert t_start >= test_missing_t[1] + self.max_delay or t_end < test_missing_t[1]-288
        else:
            assert False

        y_bar = np.zeros((tau_ahead, n_test, self.k))
        if self.d == 1:
            P_bar = np.zeros((tau_ahead, n_test, self.k))

        phi_mat = np.hstack(self.Phi)
        if self.q != 0 or self.q_s != 0:
            psi_mat = np.hstack(self.Psi)
            beta = np.concatenate((phi_mat, -psi_mat, self.Xi), axis=1)
        else:
            beta = np.concatenate((phi_mat, self.Xi), axis=1)

        u_hat = np.zeros((n_test, self.k))
        if t_start < test_missing_t[1]:
            a = self.max_delay
        else:
            a = test_missing_t[1]+self.max_delay

        for tau_i in range(tau_ahead):
            for t in range(a, t_end):
                if tau_i == 0:
                    z_data = self.make_z(0, t, z_reg_test, z_NWP_test)
                    y_vec = y_test[t-array_AR, :].flatten()
                    u_vec = u_hat[t-array_MA, :].flatten()
                    data_vec = np.hstack((y_vec, u_vec, z_data))
                    y_bar[0, t, :] = np.dot(beta, data_vec)

                    u_hat[t, :] += y_test[t, :]
                    for j, idx in enumerate(array_AR):
                        u_hat[t, :] -= np.dot(self.Phi[j], y_test[t-idx, :])
                    for i, idx in enumerate(array_MA):
                        u_hat[t, :] += np.dot(self.Psi[i], u_hat[t-idx, :])
                    u_hat[t, :] -= np.dot(self.Xi, z_data)
                else:
                    bar_AR = array_AR[tau_i-array_AR >= 0]
                    test_AR = array_AR[tau_i-array_AR < 0]
                    hat_MA = array_MA[tau_i-array_MA < 0]
                    if len(bar_AR) != 0:
                        y_vec_bar = y_bar[tau_i-bar_AR, t, :].flatten()
                    else:
                        y_vec_bar = np.array([])
                    if len(test_AR) != 0:
                        y_vec_test = y_test[t+tau_i-test_AR, :].flatten()
                    else:
                        y_vec_test = np.array([])
                    if len(hat_MA) != 0:
                        u_vec = u_hat[t+tau_i-hat_MA, :].flatten()
                    else:
                        u_vec = np.array([])
                    y_bar[tau_i, t, :] += np.dot(phi_mat, np.hstack((y_vec_bar, y_vec_test)))
                    if self.q != 0 or self.q_s != 0:
                        y_bar[tau_i, t, :] -= np.dot(psi_mat[:, (len(array_MA)-len(hat_MA))*self.k:], u_vec)
                    z_data = self.make_z(tau_i, t, z_reg_test, z_NWP_test)
                    y_bar[tau_i, t, :] += np.dot(self.Xi, z_data)

                if self.d == 1:
                    if tau_i == 0:
                        P_bar[0, t, :] = y_bar[0, t, :] + P_test[t, :]
                    else:
                        P_bar[tau_i, t, :] =  y_bar[tau_i, t, :] + P_bar[tau_i-1, t, :]
                else:
                    P_bar = y_bar

        return P_bar[tau_ahead-1, t_start-tau_ahead:t_end-tau_ahead, :]


    #@profile
    def test_newer(self, tau_ahead, test_data, P_max, P_test=None):
        """
        Same as self.test_new(), however this function uses the correct
        exogenous variables.

        Parameters
        ----------
        tau_ahead : int
            Compute tau-ahead forecast given time t-1.
        test_data : dict
            y_test : ndarray, size=(n_test, k)
                Endogenous variable.
            z_NWP_test : ndarray, size=(tau_ahead, nwp_n_test, 11*k)
                Numerical weather predictions with the given transformations.
                The first axis is the tau_ahead axis while the second axis gives
                a new set of NWP. The last axis is as follows:
                    (T, sin(WD10m), cos(WD10m), sin(WD100m), cos(WD100m),
                     WS10m, WS10m^2, WS10m^3, WS100m, WS100m^2, WS100m^3).
            z_reg_test : ndarray, size=(n_test, 2)
                Regulation data for DK1 in the first column and DK2 in the second
                column.
            test_missing_t : list
                List of time indices where a discontinuity in time is present due
                to missing power history data. The first entry in the list is
                zero and the last entry in the list is n.
        P_max : ndarray, size=(k,)
            Maximum wind power measured in the training data for each wind
            area.
        P_test : ndarray, size=(k,), optional
            Wind power at time t-2. Used when first order differencing is used.
            The default is None.

        Returns
        -------
        MSE : ndarray, size=(tau_ahead, k)
            Mean squared error of wind power forecast.
        NMAE : ndarray, size=(tau_ahead, k)
            Normalised mean absolute error of wind power forecast.
        eps : ndarray, size=(tau_ahead, n_test, k)
            Residuals of wind power forecast.
        """
        print("Commence testing...")

        y_test = test_data["y"]
        z_reg_test = test_data["z_reg"]
        z_NWP_test = test_data["z_NWP"]
        test_missing_t = test_data["missing_t"]

        array_AR = np.array([j_s*self.s+j for j_s in range(self.p_s+1) for j in range(self.p+1)])[1:]
        array_MA = np.array([i_s*self.s+i for i_s in range(self.q_s+1) for i in range(self.q+1)])[1:]

        test_nr_missing_t = len(test_missing_t)-1
        n_test, _ = y_test.shape

        y_bar = np.zeros((tau_ahead, n_test, self.k))
        if self.d == 1:
            P_bar = np.zeros((tau_ahead, n_test, self.k))

        idx_list = []

        u_hat = self.estimate_noise_newer(tau_ahead, y_test, z_NWP_test,
                                          z_reg_test, test_missing_t)

        phi_mat = np.hstack(self.Phi)
        if self.q != 0 or self.q_s != 0:
            psi_mat = np.hstack(self.Psi)
            beta = np.concatenate((phi_mat, -psi_mat, self.Xi), axis=1)
        else:
            beta = np.concatenate((phi_mat, self.Xi), axis=1)

        # Compute the tau-ahead prediction for each time in the test set
        for tau_i in range(tau_ahead):
            #if tau_i > 0:
            #    z_test = self.make_z_test(0, z_NWP_test, z_reg_test, d=d, l=l)
            if tau_i % 20 == 0:
                print("Tau ahead: {}".format(tau_i))
            for missing_t_idx in range(test_nr_missing_t):
                a = test_missing_t[missing_t_idx]+self.max_delay
                if missing_t_idx < test_nr_missing_t-1:
                    b = test_missing_t[missing_t_idx+1]-tau_ahead-288
                else:
                    b = test_missing_t[missing_t_idx+1]-tau_ahead
                for t in range(a, b):
                    if tau_i == 0:
                        idx_list.append(t)
                        y_vec = y_test[t-array_AR, :].flatten()
                        u_vec = u_hat[t-array_MA, :].flatten()
                        z_data = self.make_z(0, t, z_reg_test, z_NWP_test)
                        data_vec = np.hstack((y_vec, u_vec, z_data))
                        y_bar[0, t, :] = np.dot(beta, data_vec)
                    else:
                        bar_AR = array_AR[tau_i-array_AR >= 0]
                        test_AR = array_AR[tau_i-array_AR < 0]
                        hat_MA = array_MA[tau_i-array_MA < 0]
                        if len(bar_AR) != 0:
                            y_vec_bar = y_bar[tau_i-bar_AR, t, :].flatten()
                        else:
                            y_vec_bar = np.array([])
                        if len(test_AR) != 0:
                            y_vec_test = y_test[t+tau_i-test_AR, :].flatten()
                        else:
                            y_vec_test = np.array([])
                        if len(hat_MA) != 0:
                            u_vec = u_hat[t+tau_i-hat_MA, :].flatten()
                        else:
                            u_vec = np.array([])
                        y_bar[tau_i, t, :] += np.dot(phi_mat, np.hstack((y_vec_bar, y_vec_test)))
                        if self.q != 0 or self.q_s != 0:
                            y_bar[tau_i, t, :] -= np.dot(psi_mat[:, (len(array_MA)-len(hat_MA))*self.k:], u_vec)
                        z_data = self.make_z(tau_i, t, z_reg_test, z_NWP_test)
                        y_bar[tau_i, t, :] += np.dot(self.Xi, z_data)

                    if self.d == 1:
                        for tau_i_2 in range(tau_ahead):
                            if tau_i_2 == 0:
                                P_bar[tau_i_2, t, :] = y_bar[tau_i_2, t, :] + P_test[t, :]
                            else:
                                P_bar[tau_i_2, t, :] = y_bar[tau_i_2, t, :] + P_bar[tau_i_2-1, t, :]

        idx_array = np.array(idx_list)
        eps = np.zeros((tau_ahead, len(idx_list), self.k))
        for tau_i in range(tau_ahead):
            if self.d == 0:
                eps[tau_i, :, :] = y_bar[tau_i, idx_array, :] - y_test[idx_array+tau_i, :]
            elif self.d == 1:
                eps[tau_i, :, :] = P_bar[tau_i, idx_array, :] - P_test[idx_array+tau_i+1, :]

        MSE = np.mean(eps**2, axis=1)
        NMAE = np.mean(np.abs(eps), axis=1)/P_max
        #profile.print_stats()
        return MSE, NMAE, eps

    #@profile
    def EMD_test(self, tau_ahead, test_data, reg=True):
        """
        Same as self.test_new(), however this function uses the correct
        exogenous variables.

        Parameters
        ----------
        tau_ahead : int
            Compute tau-ahead forecast given time t-1.
        test_data : dict
            y_test : ndarray, size=(n_test, k)
                Endogenous variable.
            z_reg_test : ndarray, size=(n_test, 2)
                Regulation data for DK1 in the first column and DK2 in the second
                column.
            test_missing_t : list
                List of time indices where a discontinuity in time is present due
                to missing power history data. The first entry in the list is
                zero and the last entry in the list is n.

        Returns
        -------
        MSE : ndarray, size=(tau_ahead, k)
            Mean squared error of wind power forecast.
        eps : ndarray, size=(tau_ahead, n_test, k)
            Residuals of wind power forecast.
        """

        assert self.d == 0
        print("Commence testing...")

        y_test = test_data["y"]
        z_reg_test = test_data["z_reg"]
        test_missing_t = test_data["missing_t"]

        array_AR = np.array([j_s*self.s+j for j_s in range(self.p_s+1) for j in range(self.p+1)])[1:]
        array_MA = np.array([i_s*self.s+i for i_s in range(self.q_s+1) for i in range(self.q+1)])[1:]

        test_nr_missing_t = len(test_missing_t)-1
        n_test, _ = y_test.shape

        y_bar = np.zeros((tau_ahead, n_test, self.k))

        idx_list = []

        if reg is True:
            u_hat = self.estimate_noise_newer(tau_ahead, y_test, None,
                                              z_reg_test, test_missing_t)
        elif reg is False:
            u_hat = self.estimate_noise_newer(tau_ahead, y_test, None,
                                              z_reg_test, test_missing_t, reg)

        phi_mat = np.hstack(self.Phi)
        if self.q != 0 or self.q_s != 0:
            psi_mat = np.hstack(self.Psi)
            beta = np.concatenate((phi_mat, -psi_mat, self.Xi), axis=1)
        else:
            beta = np.concatenate((phi_mat, self.Xi), axis=1)

        # Compute the tau-ahead prediction for each time in the test set
        for tau_i in range(tau_ahead):
            #if tau_i > 0:
            #    z_test = self.make_z_test(0, z_NWP_test, z_reg_test, d=d, l=l)
            if tau_i % 20 == 0:
                print("Tau ahead: {}".format(tau_i))
            for missing_t_idx in range(test_nr_missing_t):
                a = test_missing_t[missing_t_idx]+self.max_delay
                if missing_t_idx < test_nr_missing_t-1:
                    b = test_missing_t[missing_t_idx+1]-tau_ahead-288
                else:
                    b = test_missing_t[missing_t_idx+1]-tau_ahead
                for t in range(a, b):
                    if tau_i == 0:
                        idx_list.append(t)
                        y_vec = y_test[t-array_AR, :].flatten()
                        u_vec = u_hat[t-array_MA, :].flatten()
                        if reg is True:
                            z_data = self.make_z(0, t, z_reg_test, None)
                        elif reg is False:
                            z_data = self.make_z(0, t, z_reg_test, None, reg)
                        data_vec = np.hstack((y_vec, u_vec, z_data))
                        y_bar[0, t, :] = np.dot(beta, data_vec)
                    else:
                        bar_AR = array_AR[tau_i-array_AR >= 0]
                        test_AR = array_AR[tau_i-array_AR < 0]
                        hat_MA = array_MA[tau_i-array_MA < 0]
                        if len(bar_AR) != 0:
                            y_vec_bar = y_bar[tau_i-bar_AR, t, :].flatten()
                        else:
                            y_vec_bar = np.array([])
                        if len(test_AR) != 0:
                            y_vec_test = y_test[t+tau_i-test_AR, :].flatten()
                        else:
                            y_vec_test = np.array([])
                        if len(hat_MA) != 0:
                            u_vec = u_hat[t+tau_i-hat_MA, :].flatten()
                        else:
                            u_vec = np.array([])
                        y_bar[tau_i, t, :] += np.dot(phi_mat, np.hstack((y_vec_bar, y_vec_test)))
                        if self.q != 0 or self.q_s != 0:
                            y_bar[tau_i, t, :] -= np.dot(psi_mat[:, (len(array_MA)-len(hat_MA))*self.k:], u_vec)
                        if reg is True:
                            z_data = self.make_z(tau_i, t, z_reg_test, None)
                        elif reg is False:
                            z_data = self.make_z(tau_i, t, z_reg_test, None, reg)
                        y_bar[tau_i, t, :] += np.dot(self.Xi, z_data)


        idx_array = np.array(idx_list)
        eps = np.zeros((tau_ahead, n_test, self.k))
        for tau_i in range(tau_ahead):
            eps[tau_i, idx_array, :] = y_bar[tau_i, idx_array, :] - y_test[idx_array+tau_i, :]

        MSE = np.mean(eps[:, idx_array, :]**2, axis=1)
        #profile.print_stats()
        return MSE, eps

    def estimate_noise_newer(self, tau_ahead, y_test, z_NWP_test, z_reg_test, test_missing_t, reg=True):
        """
        Parameters
        ----------
        tau_ahead : int
            Compute and test on up to tau-ahead forecasts.
        y_test : ndarray, size=(n_test, k)
            Endogenous variable.
        z_NWP_test : ndarray, size=(tau_ahead, nwp_n_test, 11*k)
            Numerical weather predictions with the given transformations.
            The first axis is the tau_ahead axis while the second axis gives
            a new set of NWP. The last axis is as follows:
                (T, sin(WD10m), cos(WD10m), sin(WD100m), cos(WD100m),
                 WS10m, WS10m^2, WS10m^3, WS100m, WS100m^2, WS100m^3).
        z_reg_test : ndarray, size=(n_test, 2)
            Regulation data for DK1 in the first column and DK2 in the second
            column.
        test_missing_t : list
            List of time indices where a discontinuity in time is present due
            to missing power history data. The first entry in the list is
            zero and the last entry in the list is n.

        Returns
        -------
        u_hat : ndarray, size=(n_test, k)
            Noise process.
        """
        array_AR = np.array([j_s*self.s+j for j_s in range(self.p_s+1) for j in range(self.p+1)])[1:]
        array_MA = np.array([i_s*self.s+i for i_s in range(self.q_s+1) for i in range(self.q+1)])[1:]

        test_nr_missing_t = len(test_missing_t)-1
        n_test, _ = y_test.shape

        u_hat = np.zeros((n_test, self.k))
        for missing_t_idx in range(test_nr_missing_t):
            a = test_missing_t[missing_t_idx]+self.max_delay
            if missing_t_idx < test_nr_missing_t-1:
                b = test_missing_t[missing_t_idx+1]-tau_ahead-288
            else:
                b = test_missing_t[missing_t_idx+1]-tau_ahead
            for t in range(a, b):
                u_hat[t, :] += y_test[t, :]
                for j, idx in enumerate(array_AR):
                    u_hat[t, :] -= np.dot(self.Phi[j], y_test[t-idx, :])
                for i, idx in enumerate(array_MA):
                    u_hat[t, :] += np.dot(self.Psi[i], u_hat[t-idx, :])
                if reg is True:
                    z_data = self.make_z(0, t, z_reg_test, z_NWP_test)
                elif reg is False:
                    z_data = self.make_z(0, t, z_reg_test, z_NWP_test, reg)
                u_hat[t, :] -= np.dot(self.Xi, z_data)
        return u_hat

    def z_NWP_index(self, tau_i, t):
        """
        Determine the indices to extract NWP data given the indices for the
        power data.

        Parameters
        ----------
        tau_i : int
            tau-ahead prediction.
        t : int
            Time index.

        Returns
        -------
        nwp_tau_i : int
            tau index for z_NWP.
        nwp_t : int
            Time index for z_NWP.
        """
        nwp_t, remainder1 = divmod(t, 36)
        nwp_tau_i, remainder2 = divmod(remainder1+tau_i, 12)
        return nwp_tau_i, nwp_t

    def make_NWP_z(self, tau_i, t, z_reg, z_NWP):
        """
        Function to make a z_data vector in a numpy array given a time
        (tau_i, t) and the exogenous variables z_reg and z_NWP as well as 
        the order of differencing d.

        Parameters
        ----------
        tau_i : int
            tau_i-ahead prediction index.
        t : int
            Time index.
        z_reg : ndarray, size=(n, 2)
            Regulations data ordered as (dk1, dk2).
        z_NWP : ndarray, size=(n, 11*k)
            Numerical weather prediction variables post transformations.

        Returns
        -------
        z_data : ndarray, size=(13*k,)
            Exogenous variable vector.
        """
        if self.d == 1:
            t += 1
        nwp_t, remainder1 = divmod(t, 36)
        nwp_tau_i, remainder2 = divmod(remainder1+tau_i, 12)
        z_data = np.zeros(self.r)
        if self.d == 0:
            if self.k == 1:
                z_data[0] = 1
                if self.l < 15:
                    z_data[1] = z_reg[t+tau_i, 0]
                else:
                    z_data[1] = z_reg[t+tau_i, 1]
                z_data[2:] = z_NWP[nwp_tau_i, nwp_t, self.l*11:(self.l+1)*11]
            elif self.k == 21:
                bias_list = [i for i in range(0, self.r, self.r_part)]
                z_data[bias_list] = np.ones(len(bias_list))
                reg1_list = [i for i in range(1, self.r_part*15, self.r_part)]
                reg2_list = [i for i in range(self.r_part*15+1, self.r, self.r_part)]
                z_data[reg1_list] = np.repeat(z_reg[t+tau_i, 0], 15)
                z_data[reg2_list] = np.repeat(z_reg[t+tau_i, 1], 6)
                nwp_list = [i for i in range(self.r) if i not in bias_list and i not in reg1_list and i not in reg2_list]
                z_data[nwp_list] = z_NWP[nwp_tau_i, nwp_t, :]
        elif self.d == 1:
            if self.k == 1:
                z_data[0] = 1
                if self.l < 15:
                    z_data[1] = z_reg[t+tau_i, 0] - z_reg[t+tau_i-1, 0]
                else:
                    z_data[1] = z_reg[t+tau_i, 1] - z_reg[t+tau_i-1, 1]
                if remainder2 == 0 and nwp_tau_i != 0:
                    z_data[2:] = z_NWP[nwp_tau_i, nwp_t, self.l*11:(self.l+1)*11] - z_NWP[nwp_tau_i-1, nwp_t, self.l*11:(self.l+1)*11]
                elif remainder2 == 0 and nwp_tau_i == 0 and remainder1 == 0:
                    z_data[2:] = z_NWP[0, nwp_t, self.l*11:(self.l+1)*11] - z_NWP[2, nwp_t-1, self.l*11:(self.l+1)*11]
                else:
                    z_data[2:] = np.zeros(11)
            elif self.k == 21:
                bias_list = [i for i in range(0, self.r, self.r_part)]
                z_data[bias_list] = np.ones(len(bias_list))
                reg1_list = [i for i in range(1, self.r_part*15, self.r_part)]
                reg2_list = [i for i in range(self.r_part*15+1, self.r, self.r_part)]
                z_data[reg1_list] = np.repeat(z_reg[t+tau_i, 0] - z_reg[t+tau_i-1, 0], 15)
                z_data[reg2_list] = np.repeat(z_reg[t+tau_i, 1] - z_reg[t+tau_i-1, 1], 6)
                nwp_list = [i for i in range(self.r) if i not in bias_list and i not in reg1_list and i not in reg2_list]
                if remainder2 == 0 and nwp_tau_i != 0:
                    z_data[nwp_list] = z_NWP[nwp_tau_i, nwp_t, :] - z_NWP[nwp_tau_i-1, nwp_t, :]
                elif remainder2 == 0 and nwp_tau_i == 0 and remainder1 == 0:
                    z_data[nwp_list] = z_NWP[0, nwp_t, :] - z_NWP[2, nwp_t-1, :]
                else:
                    z_data[nwp_list] = np.zeros(11*21)
        return z_data

    def make_other_z(self, tau_i, t, z_reg, z_NWP, is_this_test=False):
        """
        Function to make a z_data vector in a numpy array given a time
        (tau_i, t) and the exogenous variables z_reg as well as
        the order of differencing d.

        Parameters
        ----------
        tau_i : int
            tau_i-ahead prediction index.
        t : int
            Time index.
        z_reg : ndarray, size=(n, 2)
            Regulations data ordered as (dk1, dk2).
        is_this_test : bool, optional
            The test data starts at 10 am while training and validation
             starts at 0 am. Hence, for time of day a shift is 
             needed for the test data. The default is False.

        Returns
        -------
        z_data : ndarray, size=(4,)
            Exogenous variable vector.
        """
        if is_this_test is True:
            time_of_day = t-12*10
        else:
            time_of_day = t

        if self.d == 1:
            t += 1
        z_data = np.zeros(self.r)
        if self.d == 0:
            if self.k == 1:
                z_data[0] = 1
                if self.l < 15:
                    z_data[1] = z_reg[t+tau_i, 0]
                else:
                    z_data[1] = z_reg[t+tau_i, 1]
                z_data[2] = np.sin(2*np.pi*time_of_day/288)
                z_data[3] = np.cos(2*np.pi*time_of_day/288)
            elif self.k == 21:
                bias_list = [i for i in range(0, self.r, self.r_part)]
                z_data[bias_list] = np.ones(len(bias_list))
                reg1_list = [i for i in range(1, self.r_part*15, self.r_part)]
                reg2_list = [i for i in range(self.r_part*15+1, self.r, self.r_part)]
                z_data[reg1_list] = np.repeat(z_reg[t+tau_i, 0], 15)
                z_data[reg2_list] = np.repeat(z_reg[t+tau_i, 1], 6)
                time_of_day_list = [i for i in range(self.r) if i not in bias_list and i not in reg1_list and i not in reg2_list]
                z_data[time_of_day_list] = np.repeat(np.hstack((np.sin(2*np.pi*time_of_day/288), np.cos(2*np.pi*time_of_day/288))), 21)
        elif self.d == 1:
            if self.k == 1:
                z_data[0] = 1
                if self.l < 15:
                    z_data[1] = z_reg[t+tau_i, 0] - z_reg[t+tau_i-1, 0]
                else:
                    z_data[1] = z_reg[t+tau_i, 1] - z_reg[t+tau_i-1, 1]
                z_data[2] = np.sin(2*np.pi*time_of_day/288)
                z_data[3] = np.cos(2*np.pi*time_of_day/288)
            elif self.k == 21:
                bias_list = [i for i in range(0, self.r, self.r_part)]
                z_data[bias_list] = np.ones(len(bias_list))
                reg1_list = [i for i in range(1, self.r_part*15, self.r_part)]
                reg2_list = [i for i in range(self.r_part*15+1, self.r, self.r_part)]
                z_data[reg1_list] = np.repeat(z_reg[t+tau_i, 0] - z_reg[t+tau_i-1, 0], 15)
                z_data[reg2_list] = np.repeat(z_reg[t+tau_i, 1] - z_reg[t+tau_i-1, 1], 6)
                time_of_day_list = [i for i in range(self.r) if i not in bias_list and i not in reg1_list and i not in reg2_list]
                z_data[time_of_day_list] = np.repeat(np.hstack((np.sin(2*np.pi*time_of_day/288), np.cos(2*np.pi*time_of_day/288))), 21)
        return z_data

    def make_EMD_z(self, tau_i, t, z_reg, z_NWP, reg=True):
        """
        Function to make a z_data vector in a numpy array given a time
        (tau_i, t) and the exogenous variables z_reg.

        Parameters
        ----------
        tau_i : int
            tau_i-ahead prediction index.
        t : int
            Time index.
        z_reg : ndarray, size=(n, 2)
            Regulations data ordered as (dk1, dk2).

        Returns
        -------
        z_data : ndarray, size=(2,)
            Exogenous variable vector.
        """
        z_data = np.zeros(self.r)
        z_data[0] = 1
        if reg is True:
            if self.l < 15:
                z_data[1] = z_reg[t+tau_i, 0]
            else:
                z_data[1] = z_reg[t+tau_i, 1]
        return z_data

if __name__ == '__main__':
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)

    np.random.seed(49)

    Training_data = loadmat(parentdir+"/data_energinet/Training_data_TS.mat")
    Train_y = Training_data["y"]
    Train_z = Training_data["z"]

    wind_areas = ["DK1-1", "DK1-2", "DK1-3", "DK1-4", "DK1-5", "DK1-6", "DK1-7", "DK1-8", "DK1-9", "DK1-10", "DK1-11", "DK1-12", "DK1-13", "DK1-14", "DK1-15", "DK2-1", "DK2-2", "DK2-3", "DK2-4", "DK2-5", "DK2-6"]

    model = "model002"

    # =============================================================================
    # Hyperparameters
    # =============================================================================
    p = 10
    q = 0

    d = 1
    d_s = 0

    p_s = 0
    q_s = 0

    s = 288

    m = 12
    m_s = 0

    r_part = 13

    l = 0
    area = wind_areas[l]
    model_name = model+"_"+area
    save_path = "ARIMA_Results/"+model+"/"+model_name

    # =============================================================================
    # Data Retrieval
    # =============================================================================
    if d == 1:
        Power_train = np.expand_dims(Train_y[:, l], -1)
        y_train = Power_train[1:, :] - Power_train[:-1, :]

        z_train = Train_z[:, l*r_part:(l+1)*r_part]
        z_train = z_train[1:, :] - z_train[:-1, :]
        z_train[:, 0] = np.ones(np.shape(z_train)[0])

        n_train = np.shape(y_train)[0]
        used_n_train = n_train - (p_s*s + p)*2

        n_subtrain = int(used_n_train*0.8)
        n_validation = used_n_train - n_subtrain

        y_subtrain = y_train[:-n_validation, :]
        z_subtrain = z_train[:-n_validation, :]

        y_validation = y_train[-n_validation:, :]
        z_validation = z_train[-n_validation:, :]

        missing_t = Training_data["missing_t"][0]
        missing_t[1] = missing_t[1]-1
        missing_t[2] = missing_t[2]-1
        missing_t[3] = np.shape(y_subtrain)[0]
        missing_t = [missing_t]

        Training_data = {"y" : y_subtrain, "z" : z_subtrain, "missing_t" : missing_t}
    elif d == 0:
        y_train = np.expand_dims(Train_y[:, l], -1)

        z_train = Train_z[:, l*r_part:(l+1)*r_part]

        n_train = np.shape(y_train)[0]
        used_n_train = n_train - (p_s*s + p)*2

        n_subtrain = int(used_n_train*0.8)
        n_validation = used_n_train - n_subtrain

        y_subtrain = y_train[:-n_validation, :]
        z_subtrain = z_train[:-n_validation, :]

        y_validation = y_train[-n_validation:, :]
        z_validation = z_train[-n_validation:, :]

        missing_t = Training_data["missing_t"][0]
        missing_t[3] = np.shape(y_subtrain)[0]
        missing_t = [missing_t]

        Training_data = {"y" : y_subtrain, "z" : z_subtrain, "missing_t" : missing_t}


    # =============================================================================
    # Model Fit and Validation
    # =============================================================================

    mod = sVARMAX_quick_fit(p, d, q, p_s, q_s, s, m, m_s, Training_data, r_part)
    Phi, Psi, Xi, Sigma_u = mod.return_parameters()

    if d == 0:
        P_max = np.max(y_train, axis=0)
    elif d == 1:
        P_max = np.max(Power_train, axis=0)

    tau = 288*2
    if d == 1:
        MSE, NMAE, eps = mod.test(tau, y_validation, z_validation, [0, n_validation],
                                             P_max, d, Power_train[-n_validation-1:, :])
    elif d == 0:
        MSE, NMAE, eps = mod.test(tau, y_validation, z_validation, [0, n_validation], P_max)

