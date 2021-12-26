# Prediction of the Wind Power Production
This GitHub repository contains the scripts for the 9'th semester project on "Forecasting Wind Power Production" made by graduate students in Mathematical-Engineering at Aalborg University.

Authors:	Morten Stig Kaaber, Andreas Anton Andersen, and Martin Voigt Vejling

E-Mails:	{mkaabe17, aand17, mvejli17}@student.aau.dk

In this work time series data is forecasted using autoregressive moving average models, recurrent neural networks, and the empirical mode decomposition.


## Dependencies
This project is created with `Python 3.9`

Dependencies:
```
matplotlib 3.4.3
numpy 1.21.2
pandas 1.3.4
pytorch 1.10
datetime 3.9
scipy 1.7.2
```


## Data
- data_energinet/
  - EMD_Power/
    - DK1-1_SampEN.npy
    - DK1-2_SampEN.npy
    - DK1-3_SampEN.npy
    - DK1-4_SampEN.npy
    - DK1-5_SampEN.npy
    - DK1-6_SampEN.npy
    - DK1-7_SampEN.npy
    - DK1-8_SampEN.npy
    - DK1-9_SampEN.npy
    - DK1-10_SampEN.npy
    - DK1-11_SampEN.npy
    - DK1-12_SampEN.npy
    - DK1-13_SampEN.npy
    - DK1-14_SampEN.npy
    - DK1-15_SampEN.npy
    - DK2-1_SampEN.npy
    - DK2-2_SampEN.npy
    - DK2-3_SampEN.npy
    - DK2-4_SampEN.npy
    - DK2-5_SampEN.npy
    - DK2-6_SampEN.npy
  - Indicies/
  - EMD_Subtraining_Data.npy
  - EMD_Test_Data.npy
  - EMD_Training_Data.npy
  - EMD_Validation_Data.npy
  - New_Subtraining_Data.mat
  - New_Test_Data.mat
  - New_Training_Data.mat
  - New_Validation_Data.mat

## Files:
- data_energinet/
- Learning/
- Models/
- Results/
	- s-ARIMAX/
	- s-VARIMAX/
	- `Open_ARIMA_Results.py`
	- `Open_VARIMA_Results.py`
- Modules/
	- `Import_Data.py`
	- `sVARMAX_Module.py`
	- `Parallel_RNN_Module.py`
	- `Multi_RNN_Module.py`
	- `EMD_RNN.py`
- `Parallel_RNN.py`
- `Multi_RNN.py`
- `EMD_LSTM_ARMA.py`
- `sARIMAX_validation.py`
- `sARIMAX_test.py`
- `sVARIMAX_validation.py`
- `sVARIMAX_test.py`

## Modules:
`Import_Data.py`
	- Module with functionality used in Parallel_RNN and Multi_RNN to import and organise data.

`sVARMAX_Module.py`
	- Module containing the main class for estimation and forecasting with s-ARIMAX and s-VARIMAX models.

`Parallel_RNN_Module.py`
	- Module containing the methods used to fit and forecast using the univariate RNN models.

`Multi_RNN_Module.py`
	- Module containing the methods used to fit and forecast using the multivariate RNN models.

`EMD_RNN.py`
	- Module containing the methods used to fit and forecast using the EMD-LSTM-ARMA models.

## Scripts:
`Parrallel_RNN.py`
	- Script for training and testing univariate LSTMs and GRUs.
	
`Multi_RNN.py`
	- Script for training and testing multivariate LSTMs and GRUs.

`EMD_LSTM_ARMA.py`
	- Script for training and testing the EMD_LSTM_ARMA model.

`sARIMAX_validation.py`
	- Script for fitting and evaluating s-ARIMAX models using the subtraining data and the validation data.

`sARIMAX_test.py`
	- Script for fitting and evaluating s-ARIMAX models using the training data and the test data.

`sVARIMAX_validation.py`
	- Script for fitting and evaluating s-VARIMAX models using the subtraining data and the validation data.

`sVARIMAX_test.py`
	- Script for fitting and evaluating s-VARIMAX models using the training data and the test data.

`Open_ARIMA_Results.py`
	- Script for opening and displaying the results made using the s-ARIMAX models.

`Open_VARIMA_Results.py`
	- Script for opening and displaying the results made using the s-VARIMAX models.

## Usage:
To use this GitHub repository follow these steps:

1) Add the empty folders to your local repository and save the supplied data as indicated by the Files section.
2) Install Python with the dependencies stated in the Dependencies section.
3) Train and test models using scripts. Make sure to set the parameters etc. in the scripts. Note that the neural network models are compatible with GPU computing if CUDA is available and we recommend using a GPU for these computations.
