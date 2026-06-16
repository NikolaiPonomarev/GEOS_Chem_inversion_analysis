import numpy as np
import netCDF4 as nc
from pathlib import Path
import glob
import sys
from datetime import datetime, timedelta
import warnings
import xarray as xr
from functools import partial
from multiprocessing import Pool

INV_PATH = Path("/scratch/local/enkf_oco2_inv_af/oco_inv")
DATA_PATH = Path("/scratch/local/enkf_oco2_inv_af/oco2_hm_11r_sat_np/2019")
OUTPUT_PATH = Path("/exports/geos.ed.ac.uk/palmer_group/nponomar/GEOS_Chem_inversion_analysis_AF/GEOS_Chem_inversion_analysis/cost_function")

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# NE = 100
# NX = 100
# NT = 11  # Number of assimilation steps

# flux_variance = 0.5 # only if there are no correlations between flux regions, otherwise need to define the P matrix

def get_unique_dates_per_step(NT):

    step_dates = {}
    
    for step in range(NT):
        step_str = f"ST{step:03d}"
        pattern = f"hm.{step_str}.EN*-EN*.oco2_v10.*.npz"
        files = glob.glob(str(DATA_PATH / pattern))
        
        dates = set()
        for f in files:
            # Extract date from filename (format: ...oco2_v10.YYYYMMDD.npz)
            basename = Path(f).stem
            parts = basename.split('.')
            for part in parts:
                if len(part) == 8 and part.isdigit():
                    dates.add(part)
                    break
        
        step_dates[step] = sorted(list(dates))
        print(f"Step {step:02d}: {len(dates)} unique dates found")
    
    return step_dates

def read_state_vector_file(step):

    filepath = INV_PATH / "inv_err_step.nc"
    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return None, None, None, None
    print(f"Reading state vector file: {filepath}")
    with nc.Dataset(filepath, 'r') as f:
        NT = len(f.dimensions['nt'])
        NX = len(f.dimensions['nx'])
        # Prior state vector (x0)
        prior_x = f.variables['x0'][step, :] if step < len(f.dimensions['nt']) else None
        # Posterior state vector (x)
        posterior_x = f.variables['x'][step, :] if step < len(f.dimensions['nt']) else None
        # Prior flux (flux0)
        prior_flux = f.variables['flux0'][step, :] if step < len(f.dimensions['nt']) else None
        # Posterior flux (flux)
        posterior_flux = f.variables['flux'][step, :] if step < len(f.dimensions['nt']) else None
    
    return prior_x, posterior_x, prior_flux, posterior_flux, NT, NX

def read_increment_file(step):

    filepath = INV_PATH / f"inc_oco_assim_step.{step:02d}.nc"
    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return None, None, None
    print(f"Reading increment file: {filepath}")
    with nc.Dataset(filepath, 'r') as f:
        NE = len(f.dimensions['ne'])
        xtm = f.variables['xtm'][:, :] if 'xtm' in f.variables else None
        inc_m = f.variables['inc_m'][:] if 'inc_m' in f.variables else None
        xinc = f.variables['xinc'][:] if 'xinc' in f.variables else None
    
    return xtm, inc_m, xinc, NE

def read_obs_file(date_str, step=None):

    filepath = INV_PATH / f"obs_oco_assim_step.{date_str}.nc"
    if not filepath.exists():
        print(f"Warning: {filepath} not found for date {date_str}")
        return None, None, None, None, None, None, None, None
    print(f"Reading obs file: {filepath}")
    with nc.Dataset(filepath, 'r') as f:
        obs = f.variables['obs'][:] if 'obs' in f.variables else None
        mod = f.variables['mod'][:] if 'mod' in f.variables else None
        mod_adj = f.variables['mod_adj'][:] if 'mod_adj' in f.variables else None
        new_mod_adj = f.variables['new_mod_adj'][:] if 'new_mod_adj' in f.variables else None
        hm = f.variables['hm'][:, :] if 'hm' in f.variables else None
        err = f.variables['err'][:] if 'err' in f.variables else None
        lon = f.variables['lon'][:] if 'lon' in f.variables else None
        lat = f.variables['lat'][:] if 'lat' in f.variables else None
    
    return obs, mod, mod_adj, new_mod_adj, hm, err, lon, lat

def compute_jb_from_inc_m(inc_m, xtm=None):

    if inc_m is None:
        return np.nan
    return 0.5 * np.dot(inc_m, inc_m)

def compute_jo(obs, mod_prior, mod_posterior, obs_err, R_matrix=None):
    """
    Compute J_o = (y - H(x))^T R^{-1} (y - H(x))
    
    Args:
        obs: observation values (nobs,)
        mod_prior: prior model simulation H(x_b) (nobs,)
        mod_posterior: posterior model simulation H(x_a) (nobs,)
        obs_err: observation error standard deviation (nobs,)
        R_matrix: full R matrix (nobs, nobs) - if None, uses diagonal
    """
    nobs = len(obs)
    if nobs == 0:
        return np.nan, np.nan
    
    d_prior = obs - mod_prior
    d_post = obs - mod_posterior
    
    if R_matrix is not None and R_matrix.shape == (nobs, nobs):
        try:
            R_inv = np.linalg.inv(R_matrix)
            Jo_prior = np.dot(d_prior, np.dot(R_inv, d_prior))
            Jo_post = np.dot(d_post, np.dot(R_inv, d_post))
        except np.linalg.LinAlgError:
            # Fallback to diagonal
            Jo_prior = np.sum((d_prior / obs_err) ** 2)
            Jo_post = np.sum((d_post / obs_err) ** 2)
    else:
        # Diagonal R
        Jo_prior = np.sum((d_prior / obs_err) ** 2)
        Jo_post = np.sum((d_post / obs_err) ** 2)
    
    return Jo_prior, Jo_post

def process_single_date(step, date_str):

    obs, mod_prior, mod_adj, new_mod_adj, hm, obs_err, lon, lat = read_obs_file(date_str, step)
    try:
        mod_posterior = mod_prior + mod_adj + new_mod_adj
    except:
        print(f"Error computing posterior model for date {date_str} in step {step}")
        return None

    
    Jo_prior, Jo_post = compute_jo(obs, mod_prior, mod_posterior, obs_err)
    nobs = len(obs)
    
    return {
        'date': date_str,
        'nobs': nobs,
        'Jo_prior': Jo_prior,
        'Jo_post': Jo_post,
        'Jo_prior_norm': Jo_prior / nobs if nobs > 0 else np.nan,
        'Jo_post_norm': Jo_post / nobs if nobs > 0 else np.nan,
    }

def aggregate_step_cost(step, step_dates):
    
    dates = step_dates.get(step, [])
    
    prior_x, posterior_x, prior_flux, posterior_flux, _, _ = read_state_vector_file(step)
    xtm, inc_m, xinc, _ = read_increment_file(step)
    
    with Pool(processes=3) as pool:
        func = partial(process_single_date, step)
        daily_costs = pool.map(func, dates)
    
    daily_costs = [dc for dc in daily_costs if dc is not None]
    
    Jb = compute_jb_from_inc_m(inc_m) if inc_m is not None else np.nan
    
    result = {
        'step': step,
        'daily_costs': daily_costs,
        'Jb': Jb,
    }
    
    return result

def compute_full_cost_function(step_dates):
    results = []
    
    for step in sorted(step_dates.keys()):
        result = aggregate_step_cost(step, step_dates)
        if result is not None:
            results.append(result)
    
    return results

def save_results(results, step_dates, NE):
    
    all_daily = [(r['step'], d) for r in results for d in r['daily_costs']]
    #keep only the last occurence of each date (some dates may appear in multiple steps)
    unique = {d['date']: (step_num, d) for step_num, d in all_daily}
    
    daily_ds = xr.Dataset(
        {
            'assim_step': (['daily'], [unique[k][0] for k in sorted(unique.keys())]),
            'date': (['daily'], [unique[k][1]['date'] for k in sorted(unique.keys())]),
            'nobs': (['daily'], [unique[k][1]['nobs'] for k in sorted(unique.keys())]),
            'Jo_prior': (['daily'], [unique[k][1]['Jo_prior'] for k in sorted(unique.keys())]),
            'Jo_post': (['daily'], [unique[k][1]['Jo_post'] for k in sorted(unique.keys())]),
            'Jo_prior_norm': (['daily'], [unique[k][1]['Jo_prior_norm'] for k in sorted(unique.keys())]),
            'Jo_post_norm': (['daily'], [unique[k][1]['Jo_post_norm'] for k in sorted(unique.keys())]),
        }
    )

    jb_ds = xr.Dataset(
        {
            'Jb': (['step'], [r['Jb'] for r in results]),
            'step': (['step'], [r['step'] for r in results]),
        }
    )
    combined = xr.merge([daily_ds, jb_ds])
    combined.attrs['NE'] = NE
    combined.to_netcdf(OUTPUT_PATH / "cost_function_data.nc")

def main():

    _, _, _, _, NT, NX = read_state_vector_file(0)
    _, _, _, NE = read_increment_file(0)
    print(f"Inferred: NT={NT}, NX={NX}, NE={NE}")
    step_dates = get_unique_dates_per_step(NT)
    
    results = compute_full_cost_function(step_dates)
    
    save_results(results, step_dates, NE)

if __name__ == "__main__":
    main()