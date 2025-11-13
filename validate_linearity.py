import xarray as xr
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import glob
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates


outdir = '/scratch/local/for_gc_test/enkf_oco2_inv_v14/'

def compare_enkf_vs_rerun(inc_file, step_file, obs_mean_file, tol=1e-4):
    """
    Compare ENKF predicted concentrations to rerun preprocessed model at the same observation points.

    Parameters
    ----------
    inc_file : str
        Path to inc_oco_assim_step.*.nc (ENKF increments)
    step_file : str
        Path to obs_oco_assim_step.*.nc (step observations)
    obs_mean_file : str
        Path to *_obs_mean.*.nc (rerun preprocessed mod)
    tol : float
        Tolerance for matching lon, lat, time

    Returns
    -------
    enkf_pred, rerun_mod, diff : np.ndarray
        Arrays of matched ENKF predictions, rerun model values, and their difference
    """
    # print('Processing the data:\n', step_file, '\n', obs_mean_file, '\n', inc_file)
    # Load datasets
    ds_step = xr.open_dataset(step_file)
    ds_obs_mean = xr.open_dataset(obs_mean_file)
    
    # ENKF predicted concentration: mod + mod_adj + new_mod_adj
    enkf_pred_all = ds_step.mod.values + ds_step.mod_adj.values + ds_step.new_mod_adj.values
    
    # Rerun model
    rerun_mod_all = ds_obs_mean.mod.values
    
    # Coordinates
    step_lon = ds_step.lon.values
    step_lat = ds_step.lat.values
    step_time = ds_step.time.values
    
    obs_lon = ds_obs_mean.lon.values
    obs_lat = ds_obs_mean.lat.values
    obs_time = ds_obs_mean.time.values
    
    # Find matching indices
    matched_idx_step = []
    matched_idx_rerun = []
    
    for i, (lon_s, lat_s, t_s) in enumerate(zip(step_lon, step_lat, step_time)):
        # Find indices in rerun dataset close enough
        idx = np.where(
            (np.abs(obs_lon - lon_s) < tol) &
            (np.abs(obs_lat - lat_s) < tol) &
            (np.abs(obs_time - t_s) < tol)
        )[0]
        if len(idx) > 0:
            matched_idx_step.append(i)
            matched_idx_rerun.append(idx[0])  # take first match
    
    enkf_pred = enkf_pred_all[matched_idx_step]
    rerun_mod = rerun_mod_all[matched_idx_rerun]
    
    diff = enkf_pred - rerun_mod
    matched_lat = step_lat[matched_idx_step]
    
    n_points = len(diff)
    print(f"Number of matched points: {n_points}")

    if n_points > 0:
        print("Mean difference (ENKF - rerun):", np.mean(diff))
        print("Max difference:", np.max(diff))
        print("Min difference:", np.min(diff))
    else:
        print('!!! NO DATA !!!:\n', step_file, '\n', obs_mean_file, '\n', inc_file)
    
    return enkf_pred, rerun_mod, diff, matched_lat


# # Example usage
# inc_file = '/scratch/local/for_gc_test/enkf_oco2_inv_v14/oco_inv/inc_oco_assim_step.00.nc'
# step_file = '/scratch/local/for_gc_test/enkf_oco2_inv_v14/oco_inv/obs_oco_assim_step.20180101.nc'
# obs_mean_file = '/scratch/local/for_gc_test/enkf_oco2_inv_v14/oco_inv/oco2_v10_obs_mean.20180101.nc'

# enkf_pred, rerun_mod, diff = compare_enkf_vs_rerun(inc_file, step_file, obs_mean_file)

def extract_date_from_filename(fname, pattern=r'(\d{8})'):
    """
    Extract date from filename as datetime object.
    pattern: regex that matches YYYYMMDD in the filename
    """
    m = re.search(pattern, os.path.basename(fname))
    if m:
        return datetime.strptime(m.group(1), '%Y%m%d')
    else:
        return None

def get_daily_files_for_step(inc_file, step_files_pattern, obs_mean_files_pattern):
    """
    Map an assimilation step to daily files based on dates in filenames
    """
    # Load all daily files
    step_files_all = sorted(glob.glob(step_files_pattern))
    obs_mean_files_all = sorted(glob.glob(obs_mean_files_pattern))

    # Extract step index from increment file
    step_index = int(re.search(r'inc_oco_assim_step\.(\d+)', os.path.basename(inc_file)).group(1))

    # Compute step start and end dates (monthly steps starting Jan 2018)
    step_start_date = datetime(2018, 1, 1) + relativedelta(months=step_index)
    step_end_date = step_start_date + relativedelta(days=30)  # rough 30-day window

    # Filter daily files that fall in this window
    daily_step_files = [f for f in step_files_all
                        if extract_date_from_filename(f) is not None and
                        step_start_date <= extract_date_from_filename(f) <= step_end_date]

    daily_obs_mean_files = [f for f in obs_mean_files_all
                            if extract_date_from_filename(f) is not None and
                            step_start_date <= extract_date_from_filename(f) <= step_end_date]

    return daily_step_files, daily_obs_mean_files

# Path patterns
inc_files = sorted(glob.glob('/scratch/local/for_gc_test/enkf_oco2_inv_v14/oco_inv/inc_oco_assim_step.*.nc'))
step_files_pattern = '/scratch/local/for_gc_test/enkf_oco2_inv_v14/oco_inv/obs_oco_assim_step.*.nc'
obs_mean_files_pattern = '/scratch/local/for_gc_test/enkf_oco2_inv_v14/oco_inv/oco2_v10_obs_mean.*.nc'

enkf_series = []
rerun_series = []
diff_series = []
######Sequential
# # Loop over all assimilation steps and print files
# for step_idx, inc_file in enumerate(inc_files):
#     daily_step_files, daily_obs_mean_files = get_daily_files_for_step(
#         inc_file,
#         step_files_pattern,
#         obs_mean_files_pattern
#     )
#     # Sort daily files to align dates
#     daily_step_files = sorted(daily_step_files)
#     daily_obs_mean_files = sorted(daily_obs_mean_files)

#     # Loop over all daily files in this step
#     for step_file, obs_mean_file in zip(daily_step_files, daily_obs_mean_files):
#         enkf_pred, rerun_mod, diff = compare_enkf_vs_rerun(inc_file, step_file, obs_mean_file)
        
#         enkf_series.append(enkf_pred)
#         rerun_series.append(rerun_mod)
#         diff_series.append(diff)

# # Convert lists to single arrays (time x obs)
# enkf_series = np.concatenate(enkf_series)
# rerun_series = np.concatenate(rerun_series)
# diff_series = np.concatenate(diff_series)

# print("Collected time series:")
# print("ENKF:", enkf_series.shape)
# print("Rerun:", rerun_series.shape)
# print("Diff:", diff_series.shape)
######Multithreading
# ---- Function to process a single daily pair ----
def process_daily_file_pair(inc_file, step_file, obs_mean_file):
    return compare_enkf_vs_rerun(inc_file, step_file, obs_mean_file)

# # ---- Loop over assimilation steps ----
# for step_idx, inc_file in enumerate(inc_files):
#     daily_step_files, daily_obs_mean_files = get_daily_files_for_step(
#         inc_file, step_files_pattern, obs_mean_files_pattern
#     )

#     # Use processes instead of threads
#     with ProcessPoolExecutor(max_workers=4) as executor:
#         futures = [executor.submit(compare_enkf_vs_rerun, inc_file, s, o)
#                    for s, o in zip(daily_step_files, daily_obs_mean_files)]
        
#         for future in as_completed(futures):
#             enkf_pred, rerun_mod, diff = future.result()
#             enkf_series.append(enkf_pred)
#             rerun_series.append(rerun_mod)
#             diff_series.append(diff)

# # Convert lists to single arrays
# enkf_series = np.concatenate(enkf_series)
# rerun_series = np.concatenate(rerun_series)
# diff_series = np.concatenate(diff_series)

# print("Collected time series:")
# print("ENKF:", enkf_series.shape)
# print("Rerun:", rerun_series.shape)
# print("Diff:", diff_series.shape)

daily_data = []  # <-- store per-day data here

for step_idx, inc_file in enumerate(inc_files):
    daily_step_files, daily_obs_mean_files = get_daily_files_for_step(
        inc_file, step_files_pattern, obs_mean_files_pattern
    )

    # Sort daily files
    daily_step_files = sorted(daily_step_files)
    daily_obs_mean_files = sorted(daily_obs_mean_files)

    # Parallel processing 4 files at a time
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(compare_enkf_vs_rerun, inc_file, s, o): (s, o)
                   for s, o in zip(daily_step_files, daily_obs_mean_files)}

        for future in as_completed(futures):
            step_file, obs_mean_file = futures[future]
            enkf_pred, rerun_mod, diff, latitudes = future.result()
            n_obs = len(rerun_mod)
            date = extract_date_from_filename(step_file)

            daily_data.append({
                'date': date,
                'enkf': enkf_pred,
                'rerun': rerun_mod,
                'diff': diff,
                'lat': latitudes,
                'n_obs': n_obs
            })




dates = [d['date'] for d in daily_data]
enkf_mean = [np.mean(d['enkf']) if len(d['enkf'])>0 else np.nan for d in daily_data]
rerun_mean = [np.mean(d['rerun']) if len(d['rerun'])>0 else np.nan for d in daily_data]
diff_mean = [np.mean(d['diff']) if len(d['diff'])>0 else np.nan for d in daily_data]
n_obs = [d['n_obs'] for d in daily_data]
fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

# Top: model vs ENKF prediction
axes[0].plot(dates, rerun_mean, label='Rerun model', color='blue')
axes[0].plot(dates, enkf_mean, label='ENKF prediction', color='red')
axes[0].set_ylabel('Mean value')
axes[0].legend()
axes[0].set_title('Daily mean: model vs ENKF prediction')

# Middle: difference + pale points for all observations
axes[1].plot(dates, diff_mean, label='Mean difference', color='black')
for i, d in enumerate(daily_data):
    if len(d['diff'])>0:
        axes[1].scatter([d['date']]*len(d['diff']), d['diff'], color='gray', alpha=0.3, s=5)
axes[1].set_ylabel('Difference')
axes[1].set_title('Difference (ENKF - rerun) and all obs for the day')

# Bottom: number of observations
axes[2].bar(dates, n_obs, color='green', alpha=0.7)
axes[2].set_ylabel('Number of obs')
axes[2].set_title('Number of observations per day')
axes[2].set_xlabel('Date')

plt.tight_layout()
plt.savefig('Rerun_vs_EnKF_prediction.png')



from collections import Counter

# Extract dates
dates = [x['date'] for x in daily_data]

# Count occurrences
date_counts = Counter(dates)

# Check if all dates are unique
print("Number of unique dates:", len(date_counts))
print("Total entries:", len(dates))

# List dates that appear more than once
duplicates = {date: count for date, count in date_counts.items() if count > 1}
print("Duplicate dates and counts:", duplicates)


###### Take into account duplictes

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# Aggregate by date
agg_data = defaultdict(lambda: {'enkf': [], 'rerun': [], 'diff': [], 'n_obs': 0})

for d in daily_data:
    date = d['date']
    agg_data[date]['enkf'].append(d['enkf'])
    agg_data[date]['rerun'].append(d['rerun'])
    agg_data[date]['diff'].append(d['diff'])
    agg_data[date]['n_obs'] += d['n_obs']

# Compute daily means
dates_sorted = sorted(agg_data.keys())
enkf_mean = []
rerun_mean = []
diff_mean = []
n_obs = []

for date in dates_sorted:
    enkf_all = np.concatenate(agg_data[date]['enkf'])
    rerun_all = np.concatenate(agg_data[date]['rerun'])
    diff_all = np.concatenate(agg_data[date]['diff'])
    
    enkf_mean.append(np.mean(enkf_all))
    rerun_mean.append(np.mean(rerun_all))
    diff_mean.append(np.mean(diff_all))
    n_obs.append(agg_data[date]['n_obs'])

# Compute overall stats for ENKF and Rerun
enkf_overall_mean = np.mean(np.concatenate([np.concatenate(agg_data[d]['enkf']) for d in dates_sorted]))
enkf_overall_std = np.std(np.concatenate([np.concatenate(agg_data[d]['enkf']) for d in dates_sorted]))
enkf_overall_median = np.median(np.concatenate([np.concatenate(agg_data[d]['enkf']) for d in dates_sorted]))

rerun_overall_mean = np.mean(np.concatenate([np.concatenate(agg_data[d]['rerun']) for d in dates_sorted]))
rerun_overall_std = np.std(np.concatenate([np.concatenate(agg_data[d]['rerun']) for d in dates_sorted]))
rerun_overall_median = np.median(np.concatenate([np.concatenate(agg_data[d]['rerun']) for d in dates_sorted]))
# Compute overall stats for difference
diff_all_values = np.concatenate([np.concatenate(agg_data[d]['diff']) for d in dates_sorted])
diff_overall_mean = np.mean(diff_all_values)
diff_overall_std = np.std(diff_all_values)
diff_overall_median = np.median(diff_all_values)

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# Top: model vs ENKF prediction
axes[0].plot(dates_sorted, rerun_mean, label='Rerun model', color='blue')
axes[0].plot(dates_sorted, enkf_mean, label='ENKF prediction', color='red')
axes[0].set_ylabel('Mean value (ppm)')
axes[0].legend()
# axes[0].set_title('Daily mean: model vs ENKF prediction')
# Update top subplot title
axes[0].set_title(
    f'Daily mean: model vs ENKF prediction\n'
    f'ENKF mean±std: {enkf_overall_mean:.2f}±{enkf_overall_std:.2f}, median: {enkf_overall_median:.2f} | '
    f'Rerun mean±std: {rerun_overall_mean:.2f}±{rerun_overall_std:.2f}, median: {rerun_overall_median:.2f}'
)
# Middle: difference + pale points for all obs
axes[1].plot(dates_sorted, diff_mean, label='Mean difference', color='black')
for date in dates_sorted:
    diff_points = np.concatenate(agg_data[date]['diff'])
    axes[1].scatter([date]*len(diff_points), diff_points, color='gray', alpha=0.3, s=5)
axes[1].set_ylabel('Difference (ppm)')
# axes[1].set_title('Difference (ENKF - rerun) and all obs for the day')


# Update middle subplot title
axes[1].set_title(
    f'Difference (ENKF - rerun)\n'
    f'Mean±std: {diff_overall_mean:.2f}±{diff_overall_std:.2f}, median: {diff_overall_median:.2f}'
)
# Bottom: number of observations
axes[2].bar(dates_sorted, n_obs, color='green', alpha=0.7)
axes[2].set_ylabel('Number of obs')
axes[2].set_title('Number of observations per day')
axes[2].set_xlabel('Date')

plt.tight_layout()

plt.savefig(outdir + 'Rerun_vs_EnKF_prediction.png')


#####Check lat depend.
# Aggregate by date including latitudes
agg_data = defaultdict(lambda: {'enkf': [], 'rerun': [], 'diff': [], 'lat': [], 'n_obs': 0})

for d in daily_data:
    date = d['date']
    agg_data[date]['enkf'].append(d['enkf'])
    agg_data[date]['rerun'].append(d['rerun'])
    agg_data[date]['diff'].append(d['diff'])
    agg_data[date]['lat'].append(d['lat'])  # new: store latitude array
    agg_data[date]['n_obs'] += d['n_obs']

# Compute daily means (unchanged)
dates_sorted = sorted(agg_data.keys())
enkf_mean = []
rerun_mean = []
diff_mean = []
n_obs = []

for date in dates_sorted:
    enkf_all = np.concatenate(agg_data[date]['enkf'])
    rerun_all = np.concatenate(agg_data[date]['rerun'])
    diff_all = np.concatenate(agg_data[date]['diff'])
    
    enkf_mean.append(np.mean(enkf_all))
    rerun_mean.append(np.mean(rerun_all))
    diff_mean.append(np.mean(diff_all))
    n_obs.append(agg_data[date]['n_obs'])

lat_bins = {'tropical': (-23.5, 23.5), 'mid': (23.5, 66.5), 'high': (66.5, 90)}
colors = {'tropical': 'green', 'mid': 'purple', 'high': 'gold'}


fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# --- Top: model vs ENKF prediction ---
axes[0].plot(dates_sorted, rerun_mean, label='Rerun model', color='blue')
axes[0].plot(dates_sorted, enkf_mean, label='ENKF prediction', color='red')
axes[0].set_ylabel('Mean value (ppm)')
axes[0].legend()
axes[0].grid(True)
axes[0].set_title(
    f'Daily mean: model vs ENKF prediction\n'
    f'ENKF mean±std: {enkf_overall_mean:.2f}±{enkf_overall_std:.2f}, median: {enkf_overall_median:.2f} | '
    f'Rerun mean±std: {rerun_overall_mean:.2f}±{rerun_overall_std:.2f}, median: {rerun_overall_median:.2f}'
)

# --- Middle: differences with scatter points colored by latitude ---
axes[1].plot(dates_sorted, diff_mean, label='Mean difference', color='black')

# Track which bands are already labeled
added_bands = set()

for date in dates_sorted:
    diff_points = np.concatenate(agg_data[date]['diff'])
    lat_points = np.concatenate(agg_data[date]['lat'])
    
    for band, (lat_min, lat_max) in lat_bins.items():
        mask = (lat_points >= lat_min) & (lat_points <= lat_max)
        if mask.sum() > 0:
            label = band + ' latitude' if band not in added_bands else None
            axes[1].scatter(
                [date]*mask.sum(),
                diff_points[mask],
                color=colors[band],
                alpha=0.2,
                s=5,
                label=label
            )
            added_bands.add(band)  # mark as added

axes[1].set_ylabel('Difference (ppm)')
axes[1].grid(True)
axes[1].set_title(
    f'Difference (ENKF - rerun)\n'
    f'Mean±std: {diff_overall_mean:.2f}±{diff_overall_std:.2f}, median: {diff_overall_median:.2f}'
)
axes[1].legend()

# --- Bottom: number of observations ---
axes[2].bar(dates_sorted, n_obs, color='green', alpha=0.7)
axes[2].set_ylabel('Number of obs')
axes[2].grid(True)
axes[2].set_title('Number of observations per day')

# --- X-axis formatting: monthly ticks ---
axes[2].xaxis.set_major_locator(mdates.MonthLocator())
axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right')

axes[2].set_xlabel('Date')

plt.tight_layout()
plt.savefig(outdir + 'Rerun_vs_EnKF_prediction_lat_regions.png')