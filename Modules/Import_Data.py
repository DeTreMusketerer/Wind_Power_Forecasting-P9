# -*- coding: utf-8 -*-
"""

Created on Thur Nov 18 12:47:17 2021

Authors:  Andreas Anton Andersen, Martin Voigt Vejling, and Morten Stig Kaaber
E-Mails: {aand17, mvejli17, mkaabe17}@student.aau.dk

Contains functionality used to import data and handle the missing data
for the neural network models. This entails making and saving dictionaries to
the folder data_energinet/Indices. When the indices has already been made and
saved, the saved files are simply loaded into the script instead.

The module has been developed using Python 3.9 with the
libraries numpy, datetime, and pickle.

"""

import numpy as np
import datetime as dt
import pickle


def Index_dict_train_Power(intervallength):
    """
    Makes a dictionary of valid starting datapoints.

    Parameters
    ----------
    intervallength : int
        Number of hours of history used for prediction.

    Returns
    -------
    idx_dict : dict
        dictionary of valid starting datapoints.

    """
    
    try:
        with open('data_energinet/Indices/index_dict_power_train_{}.pickle'.format(intervallength), 'rb') as file:
            idx_dict = pickle.load(file)
        
    except FileNotFoundError:
        missing_data = [dt.datetime(2019,4,29,22,0,0), dt.datetime(2019,5,5,22,0,0),
                        dt.datetime(2019,4,30,21,55,0), dt.datetime(2019,5,6,21,55,0)]
        starttime = dt.datetime(2019, 1, 1, 0, 0)
        delta = dt.timedelta(hours = intervallength, minutes = 5)
        predicttime = starttime + delta
        train_end = dt.datetime(2020, 10, 5, 9, 55)
        shift = dt.timedelta(minutes = 5)
        idx_dict = {}
        idx = 0
        idx2 = 0
        while predicttime<train_end:
            if starttime <= missing_data[0] and missing_data[0] <= predicttime or missing_data[0] <= starttime and starttime <= missing_data[2]:
                starttime += shift
                predicttime += shift
                idx += 1
            elif starttime <= missing_data[1] and missing_data[1] <= predicttime or missing_data[1] <= starttime and starttime <= missing_data[3]:
                starttime += shift
                predicttime += shift
                idx += 1
            else:
                starttime += shift
                predicttime += shift
                idx_dict[idx2] = int(idx)
                idx += 1
                idx2 += 1
        
        with open('data_energinet/Indices/index_dict_power_train_{}.pickle'.format(intervallength), 'wb') as file:
            pickle.dump(idx_dict, file)
    return idx_dict


def Index_dict_validation_Power(intervallength):
    """
    Makes a dictionary of valid starting datapoints.

    Parameters
    ----------
    intervallength : int
        Number of hours of history used for prediction.

    Returns
    -------
    idx_dict : dict
        dictionary of valid starting datapoints.

    """
    
    try:
        with open('data_energinet/Indices/index_dict_power_validation_{}.pickle'.format(intervallength), 'rb') as file:
            idx_dict = pickle.load(file)
        
    except FileNotFoundError:
        starttime = dt.datetime(2020, 8, 2, 0, 0)
        delta = dt.timedelta(hours = intervallength, minutes = 5)
        predicttime = starttime + delta
        val_end = dt.datetime(2020, 10, 5, 9, 55)
        shift = dt.timedelta(minutes = 5)
        idx_dict = {}
        idx = 0
        idx2 = 0
        while predicttime<val_end:
            starttime += shift
            predicttime += shift
            idx_dict[idx2] = int(idx)
            idx += 1
            idx2 += 1
        
        with open('data_energinet/Indices/index_dict_power_validation_{}.pickle'.format(intervallength), 'wb') as file:
            pickle.dump(idx_dict, file)
    return idx_dict


def Index_dict_subtrain_Power(intervallength):
    """
    Makes a dictionary of valid starting datapoints.

    Parameters
    ----------
    intervallength : int
        Number of hours of history used for prediction.

    Returns
    -------
    idx_dict : dict
        dictionary of valid starting datapoints.

    """

    try:
        with open('data_energinet/Indices/index_dict_power_subtrain_{}.pickle'.format(intervallength), 'rb') as file:
            idx_dict = pickle.load(file)
        
    except FileNotFoundError:
        missing_data = [dt.datetime(2019,4,29,22,0,0), dt.datetime(2019,5,5,22,0,0),
                        dt.datetime(2019,4,30,21,55,0), dt.datetime(2019,5,6,21,55,0)]
        starttime = dt.datetime(2019, 1, 1, 0, 0)
        delta = dt.timedelta(hours = intervallength, minutes = 5)
        predicttime = starttime + delta
        sub_end = dt.datetime(2020, 8, 1, 23, 55)
        shift = dt.timedelta(minutes = 5)
        idx_dict = {}
        idx = 0
        idx2 = 0
        while predicttime<sub_end:
            if starttime <= missing_data[0] and missing_data[0] <= predicttime or missing_data[0] <= starttime and starttime <= missing_data[2]:
                starttime += shift
                predicttime += shift
                idx += 1
            elif starttime <= missing_data[1] and missing_data[1] <= predicttime or missing_data[1] <= starttime and starttime <= missing_data[3]:
                starttime += shift
                predicttime += shift
                idx += 1
            else:
                starttime += shift
                predicttime += shift
                idx_dict[idx2] = int(idx)
                idx += 1
                idx2 += 1
        
        with open('data_energinet/Indices/index_dict_power_subtrain_{}.pickle'.format(intervallength), 'wb') as file:
            pickle.dump(idx_dict, file)
    return idx_dict


def Index_dict_Power_Test(intervallength, predictlength):
    """
    Makes a dictionary of valid starting datapoints.

    Parameters
    ----------
    intervallength : int
        Number of hours of history used for prediction.
    predictlength : int
        Maximum number of hours we predict ahead.

    Returns
    -------
    idx_dict : dict
        dictionary of valid starting datapoints.

    """
    
    try:
        with open('data_energinet/Indices/index_dict_Power_Test_{}_{}.pickle'.format(intervallength, predictlength), 'rb') as file:
            idx_dict = pickle.load(file)
    except FileNotFoundError:
        missing_data = [dt.datetime(2021,4,2,22,0,0), dt.datetime(2021,4,3,21,55,0)]
        starttime = dt.datetime(2020, 10, 5, 12, 0)
        delta = dt.timedelta(hours = intervallength + predictlength)
        predicttime = starttime + delta
        test_end = dt.datetime(2021, 10, 5, 10, 0)
        shift = dt.timedelta(minutes = 5)
        idx_dict = {}
        idx = 0
        idx2 = 0
        while predicttime<test_end:
            if starttime <= missing_data[0] and missing_data[0] <= predicttime or missing_data[0] <= starttime and starttime <= missing_data[1]:
                starttime += shift
                predicttime += shift
                idx += 1
            else:
                starttime += shift
                predicttime += shift
                idx_dict[idx] = idx2
                idx += 1
                idx2 += 1
            
    with open('data_energinet/Indices/index_dict_Power_Test_{}_{}.pickle'.format(intervallength, predictlength), 'wb') as file:
        pickle.dump(idx_dict, file)
    return idx_dict


def Index_dict_train_NWP(intervallength):
    """
    Makes a dictionary of NWPs correspoding to the power starting points.

    Parameters
    ----------
    intervallength : int
        Number of hours of history used for prediction.

    Returns
    -------
    idx_dict : dict
        dictionary of corresponding NWPs to power startingpoints.

    """
    
    try:
        with open('data_energinet/Indices/index_dict_NWP_train_{}.pickle'.format(intervallength), 'rb') as file:
            idx_dict = pickle.load(file)
        
    except FileNotFoundError:
        missing_data = [dt.datetime(2019,4,29,22,0,0), dt.datetime(2019,5,5,22,0,0),
                        dt.datetime(2019,4,30,21,55,0), dt.datetime(2019,5,6,21,55,0)]
        starttime = dt.datetime(2019, 1, 1, 0, 0)
        delta = dt.timedelta(hours = intervallength, minutes = 5)
        predicttime = starttime + delta
        train_end = dt.datetime(2020, 10, 5, 9, 55)
        shift = dt.timedelta(minutes = 5)
        idx_dict = {}
        idx = 0
        t = intervallength * 12 +1
        while predicttime<train_end:
            if starttime <= missing_data[0] and missing_data[0] <= predicttime or missing_data[0] <= starttime and starttime <= missing_data[2]:
                starttime += shift
                predicttime += shift
                t += 1
            elif starttime <= missing_data[1] and missing_data[1] <= predicttime or missing_data[1] <= starttime and starttime <= missing_data[3]:
                starttime += shift
                predicttime += shift
                t += 1
            else:
                starttime += shift
                predicttime += shift
                nwp_t, rem_1 = np.divmod(t,36)
                nwp_tau, _ = np.divmod(rem_1,12)
                idx_dict[idx] = np.array([nwp_tau, nwp_t], dtype = 'int')
                idx += 1
                t += 1
                
        with open('data_energinet/Indices/index_dict_NWP_train_{}.pickle'.format(intervallength), 'wb') as file:
            pickle.dump(idx_dict, file)
    return idx_dict


def Index_dict_validation_NWP(intervallength):
    """
    Makes a dictionary of NWPs correspoding to the power starting points.

    Parameters
    ----------
    intervallength : int
        Number of hours of history used for prediction.

    Returns
    -------
    idx_dict : dict
        dictionary of corresponding NWPs to power startingpoints.

    """
    
    try:
        with open('data_energinet/Indices/index_dict_NWP_validation_{}.pickle'.format(intervallength), 'rb') as file:
            idx_dict = pickle.load(file)
        
    except FileNotFoundError:
        starttime = dt.datetime(2020, 8, 2, 0, 0)
        delta = dt.timedelta(hours = intervallength, minutes = 5)
        predicttime = starttime + delta
        val_end = dt.datetime(2020, 10, 5, 9, 55)
        shift = dt.timedelta(minutes = 5)
        idx_dict = {}
        idx = 0
        t = intervallength * 12 +1
        while predicttime<val_end:
            starttime += shift
            predicttime += shift
            nwp_t, rem_1 = np.divmod(t,36)
            nwp_tau, _ = np.divmod(rem_1,12)
            idx_dict[idx] = np.array([nwp_tau, nwp_t], dtype = 'int')
            idx += 1
            t += 1
        
        with open('data_energinet/Indices/index_dict_NWP_validation_{}.pickle'.format(intervallength), 'wb') as file:
            pickle.dump(idx_dict, file)
    return idx_dict


def Index_dict_subtrain_NWP(intervallength):
    """
    Makes a dictionary of NWPs correspoding to the power starting points.

    Parameters
    ----------
    intervallength : int
        Number of hours of history used for prediction.

    Returns
    -------
    idx_dict : dict
        dictionary of corresponding NWPs to power startingpoints.

    """
    
    try:
        with open('data_energinet/Indices/index_dict_NWP_subtrain_{}.pickle'.format(intervallength), 'rb') as file:
            idx_dict = pickle.load(file)
    except FileNotFoundError:
        missing_data = [dt.datetime(2019,4,29,22,0,0), dt.datetime(2019,5,5,22,0,0),
                        dt.datetime(2019,4,30,21,55,0), dt.datetime(2019,5,6,21,55,0)]
        starttime = dt.datetime(2019, 1, 1, 0, 0)
        delta = dt.timedelta(hours = intervallength, minutes = 5)
        predicttime = starttime + delta
        subtrain_end = dt.datetime(2020, 8, 1, 23, 55)
        shift = dt.timedelta(minutes = 5)
        idx_dict = {}
        idx = 0
        t = intervallength * 12 +1
        
        while predicttime<subtrain_end:
            if starttime <= missing_data[0] and missing_data[0] <= predicttime or missing_data[0] <= starttime and starttime <= missing_data[2]:
                starttime += shift
                predicttime += shift
                t += 1
            elif starttime <= missing_data[1] and missing_data[1] <= predicttime or missing_data[1] <= starttime and starttime <= missing_data[3]:
                starttime += shift
                predicttime += shift
                t += 1
            else:
                starttime += shift
                predicttime += shift
                nwp_t, rem_1 = np.divmod(t,36)
                nwp_tau, _ = np.divmod(rem_1,12)
                idx_dict[idx] = np.array([nwp_tau, nwp_t], dtype = 'int')
                idx += 1
                t += 1
    with open('data_energinet/Indices/index_dict_NWP_subtrain_{}.pickle'.format(intervallength), 'wb') as file:
        pickle.dump(idx_dict, file)
    return idx_dict


def Index_dict_NWP_Test(intervallength, predictlength, batch_size):
    """
    Makes a dictionary of NWPs correspoding to the power starting points.

    Parameters
    ----------
    intervallength : int
        Number of hours of history used for prediction.
    predictlength : int
        Number of hours of history used for prediction.
    batch_size : int
        Batch size.

    Returns
    -------
    idx_dict : dict
        dictionary of corresponding NWPs to power startingpoints.

    """
    
    try:
        with open('data_energinet/Indices/index_dict_NWP_Test_{}_{}_{}.pickle'.format(intervallength, predictlength, batch_size), 'rb') as file:
            idx_dict = pickle.load(file)
    except FileNotFoundError:
        missing_data = [dt.datetime(2021,4,2,22,0,0), dt.datetime(2021,4,3,21,55,0)]
        starttime = dt.datetime(2020, 10, 5, 12, 0)
        delta = dt.timedelta(hours = intervallength + predictlength)
        predicttime = starttime + delta
        test_end = dt.datetime(2021, 10, 5, 10, 0)
        shift = dt.timedelta(minutes = 5)
        idx_dict = {}
        idx_dict_start = {}
        idx = 0
        t = intervallength * 12 +1
        
        while predicttime<test_end:
            if starttime <= missing_data[0] and missing_data[0] <= predicttime or missing_data[0] <= starttime and starttime <= missing_data[1]:
                starttime += shift
                predicttime += shift
                t += 1
            else:
                starttime += shift
                predicttime += shift
                nwp_t, rem_1 = np.divmod(t,36)
                nwp_tau, _ = np.divmod(rem_1,12)
                idx_dict_start[idx] = np.array([nwp_tau, nwp_t], dtype = 'int')
                idx += 1
                t += 1
    
        antal, remington = np.divmod(len(idx_dict_start),batch_size)
        for j in range(antal):
            insert_array = np.zeros((batch_size,2), dtype = 'int')
            for i in range(batch_size):
                insert_array[i,:] = idx_dict_start[j*batch_size+i]
            idx_dict[j] = insert_array
        
        if remington != 0:
            last_array = np.zeros((remington,2), dtype = 'int')
            for i in range(remington):
                last_array[i] = idx_dict_start[antal*batch_size + i]
    
            idx_dict[j+1] = last_array
            
    with open('data_energinet/Indices/index_dict_NWP_Test_{}_{}_{}.pickle'.format(intervallength, predictlength, batch_size), 'wb') as file:
        pickle.dump(idx_dict, file)
    return idx_dict




