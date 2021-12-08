# Prediction of the Electrical Power Network
This GitHub repository contains code for the 9'th semester project on "Prediction of the Electrical Power Network" made by graduate students in Mathematical-Engineering at Aalborg University.

Authors:	Morten Stig Kaaber, Andreas Anton Andersen, and Martin Voigt Vejling

E-Mails:	{mkaabe17, aand17, mvejli17}@student.aau.dk

In this work time series data is forecasted.


## Dependencies
This project is created with `Python 3.9`

Dependencies:
```
Python 3.9
matplotlib 3.4.3
numpy 1.21.2
PyEMD 0.2.13
pyarrow 5.0.0
pandas 1.3.4
pytourch 1.9
datetime 3.9
scikit-learn 1.0.1
```

## Files
The code consists of a number of scripts and modules including dependencies between the scripts. In the following, an overview of the included files and folders will be given.

- data_py/
	- 'OpenData.py' can be used to open and visualise the data.
	- 'OpenDataUtilities.py' is a module containing functionality to open the data.
	- 'Organise_units.py' is used to form the 'units_full_info.csv' file.
	- 'Organise_regulering.py' is used to form the 'regulering.csv' file.
	- 'Wind_mill_plot.py' is used for plotting the position of wind mills on top of a map of Denmark.
	- 'Transform_coordinates.py' is a module used to transform between coordinate systems.
	- 'Area_midpoints_and_NWP_grid.py' is used to compute the midpoints of the areas and check which NWP gridpoint is the closest to each area midpoint.


## Data
- data_energinet/
  - vejrdata_csv/ (main)
  - vindfarmdata/
    - regulering_matrix/
    - scada_onshore_wind_production/
  - plants.csv
  - units_new.csv
  - Wind_Power.csv (main)
  - wind_areas_new.mat
  - units_full_info.csv
  - regulering.csv (main)
  - area_NWP.csv (main)


## Usage

