# GEOS_Chem_inversion_analysis
A collection of Python scripts for analyzing and post-processing results from GEOS-Chem EnKF inversion experiments.

### Current scripts

- **`validate_linearity.py`**  
  Compares ENKF-predicted concentrations with rerun model results at observation locations and times. Example output figure showing daily differences between ENKF and rerun simulations is available in `Examples/Rerun_vs_EnKF_prediction_lat_regions.png`. 
