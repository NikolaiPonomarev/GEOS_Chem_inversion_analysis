# GEOS_Chem_inversion_analysis
A collection of Python scripts for analyzing and post-processing results from GEOS-Chem EnKF inversion experiments.

### Current scripts

- **`validate_linearity.py`**  
  Compares ENKF-predicted concentrations with rerun model results at observation locations and times. Example output figure showing daily differences between ENKF and rerun simulations is available in `Examples/Rerun_vs_EnKF_prediction_lat_regions.png`. *Note:* Apparently rerun data is not saved, only the last step of model simulations with optimized fluxes is stored. So the test would need to be done **directly** by scaling flux changes and comparing concentration changes of the ensemble.

- **`Aggregate_XCO2_to_model_grid.py`**  
  Aggregates satellite-sampled XCO₂ data onto the GEOS-Chem model regular grid for further analysis.

- **`Aggregate_LC.py`**  
  Reprojects high-resolution land cover data to the model grid and generates coarse-grid summaries for use in inversion experiments.

- **Plot_region_flux_annual_maps.py**  
Computes and plots annual CO₂ flux maps for prior and posterior inversion results. Generates global maps of prior, posterior, and absolute/percent flux differences.

- **Prior_posterior_XCO2_map_plots.py**  
Generates maps and annual statistics for prior vs posterior XCO₂ concentrations. Produces bar plots showing bias, RMSE, CRMSE, correlation, and Taylor diagrams to evaluate inversion performance.
 
