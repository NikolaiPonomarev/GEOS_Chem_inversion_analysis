import xarray as xr
import numpy as np
import pandas as pd
import os
import glob
from multiprocessing import Pool
# ----------------------
# INPUTS
# ----------------------
grid_file = "/exports/geos.ed.ac.uk/palmer_group/run_test_2x25/OutputDir/GEOSChem.SatDiagn.20210101_0000z.nc4"
step_files = sorted(glob.glob("/scratch/local/for_gc_test/enkf_oco2_inv_v14/oco_inv/obs_oco_assim_step.*.nc"))
obs_mean_files = sorted(glob.glob("/scratch/local/for_gc_test/enkf_oco2_inv_v14/oco_inv/oco2_v*_obs_mean.*.nc"))
outdir = "/scratch/local/for_gc_test/enkf_oco2_inv_v14/oco_inv/aggregated"
os.makedirs(outdir, exist_ok=True)

# ----------------------
# LOAD MODEL GRID
# ----------------------
ds_grid = xr.open_dataset(grid_file)
lon_grid = ds_grid.lon.values
lat_grid = ds_grid.lat.values
ds_grid.close()

dlat = np.diff(lat_grid).mean()
dlon = np.diff(lon_grid).mean()
lat_edges = np.concatenate(([lat_grid[0] - dlat/2], (lat_grid[:-1] + lat_grid[1:])/2, [lat_grid[-1] + dlat/2]))
lon_edges = np.concatenate(([lon_grid[0] - dlon/2], (lon_grid[:-1] + lon_grid[1:])/2, [lon_grid[-1] + dlon/2]))

n_lat = len(lat_grid)
n_lon = len(lon_grid)

# ----------------------
# FUNCTION TO PROCESS ONE DAY
# ----------------------
# def process_day(step_file, obs_mean_file, tol=1e-4):    
#     ds_step = xr.open_dataset(step_file)
#     ds_obs_mean = xr.open_dataset(obs_mean_file)
#     time_tol = 120
#     # Match points present in assimilation
#     matched_idx = []
#     for i, (lon_s, lat_s, t_s) in enumerate(zip(ds_step.lon.values, ds_step.lat.values, ds_step.time.values)):
#         idx = np.where(
#             (np.abs(ds_obs_mean.lon.values - lon_s) < tol) &
#             (np.abs(ds_obs_mean.lat.values - lat_s) < tol) &
#             (np.abs(ds_obs_mean.time.values - t_s) < time_tol)
#         )[0]
#         if len(idx) > 0:
#             matched_idx.append(i)

#     enkf_lon = ds_step.lon.values[matched_idx]
#     enkf_lat = ds_step.lat.values[matched_idx]
#     obs = ds_step.obs.values[matched_idx]
#     rerun_mod = ds_step.mod.values[matched_idx] + ds_step.mod_adj.values[matched_idx] + ds_step.new_mod_adj.values[matched_idx]
#     # Construct optimized values by summing all relevant adjustments
#     prior_model = []
#     for lon_s, lat_s, t_s in zip(enkf_lon, enkf_lat, ds_step.time.values[matched_idx]):
#         idx = np.where(
#             (np.abs(ds_obs_mean.lon.values - lon_s) < tol) &
#             (np.abs(ds_obs_mean.lat.values - lat_s) < tol) &
#             (np.abs(ds_obs_mean.time.values - t_s) < time_tol)
#         )[0][0]
#         prior_model_value = (
#             ds_obs_mean.mod.values[idx]
#         )
#         prior_model.append(prior_model_value)
#     prior_model = np.array(prior_model)
#     # convert step times to pandas datetime (seconds since epoch)
#     dates = pd.to_datetime(ds_step.time.values, unit='s')

#     # make a pandas Series of dates only
#     date_series = pd.Series(dates.date)

#     # pick the most frequent date
#     most_common_date = pd.Timestamp(date_series.value_counts().idxmax())
#     n_points = len(enkf_lon)

#     print(f"Number of matched points: {n_points}", flush=True)    
#     print(step_file, obs_mean_file, most_common_date, flush=True)
#     ds_step.close()
#     ds_obs_mean.close()


#     return {
#         "date": most_common_date,
#         "lon": enkf_lon,
#         "lat": enkf_lat,
#         "obs": obs,
#         "prior": prior_model,
#         "optimized": rerun_mod
#     }

def process_day(step_file):
    """
    Process a single obs_oco_assim_step file and extract:
    - obs
    - prior
    - posterior
    - lon/lat/time
    """
    
    ds = xr.open_dataset(step_file)

    lon  = ds.lon.values
    lat  = ds.lat.values
    time = ds.time.values

    obs  = ds.obs.values
    prior = ds.mod.values

    # Developer's definition of posterior:
    posterior = ds.mod.values + ds.mod_adj.values + ds.new_mod_adj.values

    # Convert time (seconds since epoch) to a single date
    dates = pd.to_datetime(time, unit="s")
    date_only = pd.Series(dates.date)
    most_common_date = pd.Timestamp(date_only.value_counts().idxmax())

    ds.close()
    print('Finished reading ', step_file)
    return {
        "date": most_common_date,
        "lon": lon,
        "lat": lat,
        "obs": obs,
        "prior": prior,
        "optimized": posterior
    }


# ----------------------
# READ FILES IN PARALLEL
# ----------------------
# args_list = list(zip(step_files, obs_mean_files))
    
# with Pool(processes=4) as pool:
#     results = pool.starmap(process_day, args_list)

import re

def get_date_from_filename(fname):
    m = re.search(r'\d{8}', os.path.basename(fname))
    return m.group(0) if m else None

# # dict of obs_mean files keyed by date
# obs_mean_dict = {get_date_from_filename(f): f for f in obs_mean_files}

# # create args list only for step files that have a matching obs_mean file
# args_list = []
# for step_file in step_files:
#     step_date = get_date_from_filename(step_file)
#     obs_mean_file = obs_mean_dict.get(step_date)
#     if obs_mean_file is not None:
#         args_list.append((step_file, obs_mean_file))
#     else:
#         print(f"No obs_mean file for step {step_file}")

# # process in parallel
# with Pool(processes=4) as pool:
#     results = pool.starmap(process_day, args_list)

with Pool(processes=4) as pool:
        results = pool.map(process_day, step_files)


# ----------------------
# DAILY AGGREGATION ON MODEL GRID (using lon_bnds / lat_bnds)
# ----------------------
# extract the dates (already most frequent per file)
# dates = [pd.to_datetime(get_date_from_filename(f), format='%Y%m%d') for f, _ in args_list]
dates = [pd.to_datetime(get_date_from_filename(f), format='%Y%m%d') for f in step_files]
n_days = len(dates)

obs_daily = np.zeros((n_days, n_lat, n_lon))
prior_daily = np.zeros((n_days, n_lat, n_lon))
optimized_daily = np.zeros((n_days, n_lat, n_lon))
count_daily = np.zeros((n_days, n_lat, n_lon))

# use lon_bnds / lat_bnds from the grid
lon_bnds = ds_grid.lon_bnds.values  # shape (n_lon, 2)
lat_bnds = ds_grid.lat_bnds.values  # shape (n_lat, 2)

for i, r in enumerate(results):
    # find the grid cell index for each lon/lat
    lon_idx = np.searchsorted(lon_bnds[:, 1], r["lon"])
    lat_idx = np.searchsorted(lat_bnds[:, 1], r["lat"])

    # clip to valid indices (0..n_lon-1, 0..n_lat-1)
    lon_idx = np.clip(lon_idx, 0, n_lon-1)
    lat_idx = np.clip(lat_idx, 0, n_lat-1)

    # accumulate values
    np.add.at(obs_daily[i], (lat_idx, lon_idx), r["obs"])
    np.add.at(prior_daily[i], (lat_idx, lon_idx), r["prior"])
    np.add.at(optimized_daily[i], (lat_idx, lon_idx), r["optimized"])
    np.add.at(count_daily[i], (lat_idx, lon_idx), 1)

# average where counts > 0
mask = count_daily > 0
obs_daily[mask] /= count_daily[mask]
prior_daily[mask] /= count_daily[mask]
optimized_daily[mask] /= count_daily[mask]

# fill empty cells with NaN
obs_daily[count_daily == 0] = np.nan
prior_daily[count_daily == 0] = np.nan
optimized_daily[count_daily == 0] = np.nan

# ----------------------
# SAVE TO NETCDF
# ----------------------
out_nc = os.path.join(outdir, "daily_binned_prior_optimized_obs.nc")
ds_out = xr.Dataset(
    {
        "obs": (("time", "lat", "lon"), obs_daily),
        "prior": (("time", "lat", "lon"), prior_daily),
        "optimized": (("time", "lat", "lon"), optimized_daily),
        "counts": (("time", "lat", "lon"), count_daily)
    },
    coords={
        "time": dates,
        "lat": lat_grid,
        "lon": lon_grid
    }
)
ds_out.to_netcdf(out_nc)
print(f"Saved daily binned prior, optimized, and obs to {out_nc}")