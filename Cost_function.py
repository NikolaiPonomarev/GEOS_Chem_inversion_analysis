import numpy as np
import xarray as xr
import pandas as pd
import glob
import os
import multiprocessing as mp
inv_err_path = "/scratch/local/for_gc_test/enkf_oco2_inv_v14/oco_inv/inv_err_step.nc"
obs_pattern  = "/scratch/local/for_gc_test/enkf_oco2_inv_v14/oco_inv/obs_oco_assim_step.*.nc"
output_csv   = "cost_function_and_chi2_all_months.csv"

# Load monthly state vectors and covariances
ds_p = xr.open_dataset(inv_err_path)
nt = ds_p.dims["nt"]
nx = ds_p.dims["nx"]

x0_var = ds_p["x0"].values  # (nt, nx)
x_var  = ds_p["x"].values   # (nt, nx)
B_var  = ds_p["xerr0"].values  # (nt, nx, nx)

# Find all obs files
obs_files = sorted(glob.glob(obs_pattern))
inv_start_year = 2018
# Function to get month index (0..35) from obs file name
def get_month_index(obs_file):
    # Extract YYYYMMDD from filename, e.g., obs_oco_assim_step.20180102.nc
    base = os.path.basename(obs_file)
    date_str = base.split(".")[-2]  # '20180102'
    year = int(date_str[:4])
    month = int(date_str[4:6])
    # ENKS months start at 2018-01 = index 0
    return (year - inv_start_year) * 12 + (month - 1)

# Group obs files by month
groups = {i: [] for i in range(nt)}
for f in obs_files:
    month_idx = get_month_index(f)
    if 0 <= month_idx < nt:
        groups[month_idx].append(f)

def process_month(t):
    files = groups[t]
    if not files:
        return None
    
    y_list, Yb_list, Ya_list, Hm_list = [], [], [], []

    for f in files:
        ds = xr.open_dataset(f)
        obs = ds["obs"].values.astype(float)
        mod  = ds["mod"].values.astype(float)
        mod_adj = ds.get("mod_adj", np.zeros_like(mod))
        new_mod_adj = ds.get("new_mod_adj", np.zeros_like(mod))
        
        Yb = mod# + mod_adj
        Ya = mod + mod_adj + new_mod_adj

        y_list.append(obs)
        Yb_list.append(Yb)
        Ya_list.append(Ya)
        Hm_list.append(ds["hm"].values.astype(float))
        ds.close()

    y  = np.concatenate(y_list)
    Hx_b = np.concatenate(Yb_list)
    Hx_a = np.concatenate(Ya_list)
    Hm = np.concatenate(Hm_list, axis=0)

    xb = x0_var[t, :]
    xa = x_var[t, :]
    B  = B_var[t, :, :]

    dx = xa - xb
    Jb = 0.5 * dx @ np.linalg.solve(B, dx) / len(y)

    sigma_squared = 0.8**2 + 1.5**2
    d_prior = y - Hx_b
    d_post  = y - Hx_a

    Jo_prior = 0.5 * np.sum(d_prior**2 / sigma_squared) / len(y)
    Jo_post  = 0.5 * np.sum(d_post**2 / sigma_squared) / len(y)

    ne = Hm.shape[1]
    HPH = (Hm @ Hm.T) / (ne - 1)
    Sig = np.diag(np.full(len(y), sigma_squared)) + HPH
    chi2_full = (d_post @ np.linalg.solve(Sig, d_post)) / len(y)
    print(f"Month {t}: J_total_prior={Jb+Jo_prior:.3e}, J_total_post={Jb+Jo_post:.3e}, chi2_full={chi2_full:.3e}, nobs={len(y)}")
    return {
        "month_index": t,
        "Jb": Jb,
        "Jo_prior": Jo_prior,
        "J_total_prior": Jb + Jo_prior,
        "Jo_post": Jo_post,
        "J_total_post": Jb + Jo_post,
        "chi2_full": chi2_full,
        "nobs": len(y)
    }

# Run in parallel
with mp.Pool(processes=4) as pool:  # adjust number of processes
    results_list = pool.map(process_month, range(nt))

# Remove None entries for months with no files
results = [r for r in results_list if r is not None]



###Make some plots
import cftime
import pandas as pd
import matplotlib.pyplot as plt

# Convert results list to DataFrame for convenience
df = pd.DataFrame(results)

# Create a simple monthly time axis starting from Jan 2018
time = xr.cftime_range(
    start=f"{inv_start_year}-01-01", 
    periods=len(df), 
    freq="MS"  # Month Start
)
time = [pd.Timestamp(str(t)) if isinstance(t, cftime.DatetimeGregorian) else t for t in time]
chi2_mean = df["chi2_full"].mean()
# Figure 1: chi2_full vs month
plt.figure(figsize=(6, 5))
plt.plot(time, df["chi2_full"], marker='o', linestyle='-', label=f"chi2 (mean={chi2_mean:.2f})")
plt.title("Chi-squared per month")
plt.xlabel("Month")
plt.ylabel("Chi2")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/home/nponomar/GEOS_Chem_inversion_analysis/Examples/Chi2_values_post_v3.png', 
            dpi=300, bbox_inches='tight')

# Figure 2: J_total prior vs posterior
J_prior_mean = df["J_total_prior"].mean()
J_post_mean  = df["J_total_post"].mean()
plt.figure(figsize=(6, 5))
plt.plot(time, df["J_total_prior"], marker='o', linestyle='-', label=f"J_total_prior (mean={J_prior_mean:.2f})")
plt.plot(time, df["J_total_post"], marker='s', linestyle='--', label=f"J_total_post (mean={J_post_mean:.2f})")
plt.title("Cost function per month")
plt.xlabel("Month")
plt.ylabel("J_total")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/home/nponomar/GEOS_Chem_inversion_analysis/Examples/Cost_function_monthly_mod_only.png', 
            dpi=300, bbox_inches='tight')