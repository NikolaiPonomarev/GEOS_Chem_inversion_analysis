import os
import glob
import numpy as np
import numpy.linalg as nlg
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh
import pandas as pd 
import matplotlib.dates as mdates
from my_mapping_utils import apply_region_mapping, apply_region_mapping_matrix
import xarray as xr
# ---------- USER PATHS (change if needed) ----------
INV_ERR_FPATH = "/scratch/local/for_gc_test/enkf_oco2_inv_v14/oco_inv/inv_err_step.nc"
INC_STEP_GLOB = "/scratch/local/for_gc_test/enkf_oco2_inv_v14/oco_inv/inc_oco_assim_step*.nc"
OUTPUT = "/home/nponomar/GEOS_Chem_inversion_analysis/Examples"

first_step_date = '2018-01-01'

# ---------- helpers ----------
def safe_pinv(A, rcond=1e-8):
    """Robust pseudoinverse wrapper."""
    try:
        return nlg.pinv(A, rcond=rcond)
    except Exception as e:
        print("pinv failed:", e)
        return nlg.pinv(A)  # best effort

def participation_ratio(eigs):
    s = np.sum(eigs)
    ssq = np.sum(eigs**2)
    if ssq <= 0: 
        return 0.0
    return (s*s)/ssq

def ensure_square(A):
    return A.shape[0] == A.shape[1]

# ---------- READ inv_err_step.nc ----------
if not os.path.exists(INV_ERR_FPATH):
    raise FileNotFoundError(f"inv_err file not found: {INV_ERR_FPATH}")

ds = Dataset(INV_ERR_FPATH)
print("inv_err variables:", list(ds.variables.keys()))
# Expected names: nt, nx, ny, flux0, err0, x0, xerr0, flux, err, x, xerr
err0 = ds.variables['err0'][:]   # shape (nt, nx, ny)
err  = ds.variables['err'][:]    # shape (nt, nx, ny)
flux0 = ds.variables['flux0'][:] if 'flux0' in ds.variables else None
flux  = ds.variables['flux'][:] if 'flux' in ds.variables else None
nt, nx_e, ny_e = err0.shape
print(f"Loaded inv_err_step: nt={nt}, nx={nx_e}, ny={ny_e}")
# verify squares
if nx_e != ny_e:
    print("WARNING: err0 is not square in state dims: nx != ny. Will try to use diagonals and pinv on reshaped blocks.")

# ---------- Find inc step files ----------
inc_files = sorted(glob.glob(INC_STEP_GLOB))
print("Found inc files (sample):", inc_files[:5])
# We'll read one representative inc file per step if exists. We may have many steps.
inc_sample = inc_files[0] if inc_files else None

# ---------- Read inc sample if exists ----------
xtm_dict = {}    # map step -> xtm
incm_dict = {}   # map step -> inc_m (state-space increment or ensemble-space; we'll detect)
xinc_dict = {}   # map step -> xinc (state increment)

if inc_sample:
    for fn in inc_files:
        try:
            d = Dataset(fn)
            # attempt to extract step index from filename
            base = os.path.basename(fn)
            # naive find two-digit step
            try:
                step_str = "".join([s for s in base if s.isdigit()])
                step_int = int(step_str[-2:]) if len(step_str)>=2 else None
            except:
                step_int = None
            # read available variables
            _xtm = d.variables['xtm'][:] if 'xtm' in d.variables else None
            _inc_m = d.variables['inc_m'][:] if 'inc_m' in d.variables else None
            _xinc = d.variables['xinc'][:] if 'xinc' in d.variables else None
            key = step_int if step_int is not None else fn
            xtm_dict[key] = _xtm
            incm_dict[key] = _inc_m
            xinc_dict[key] = _xinc
            d.close()
        except Exception as e:
            print("Failed reading", fn, ":", e)

# ---------- Diagnostic 1: DOFS (state) time series ----------
dofs_time = np.full(nt, np.nan)
for t in range(nt):
    Pa = err0[t,:,:]
    Pp = err[t,:,:]
    # ensure symmetric
    Pa = 0.5*(Pa + Pa.T)
    Pp = 0.5*(Pp + Pp.T)
    # if shapes mismatch or not square, fall back to diagonal-based approx or pinv on submatrix
    if Pa.shape[0] != Pa.shape[1] or not ensure_square(Pa):
        # fallback: DOFS approx from trace diag comparison
        diag_pa = np.diag(Pa) if Pa.shape[0]==Pa.shape[1] else np.diag(Pa[:min(Pa.shape)])
        diag_pp = np.diag(Pp) if Pp.shape[0]==Pp.shape[1] else np.diag(Pp[:min(Pp.shape)])
        # avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            fvr = 1.0 - np.nan_to_num(diag_pp/diag_pa, nan=0.0, posinf=0.0, neginf=0.0)
        dofs_time[t] = np.sum(fvr)  # crude sum of fractional reductions as proxy
    else:
        Pa_inv = safe_pinv(Pa)
        A = np.eye(Pa.shape[0]) - Pp @ Pa_inv
        dofs_time[t] = np.trace(A)

# Assign date to each timestep
nt = len(dofs_time)
start = pd.to_datetime(first_step_date)

# build dates: start, start +1 month, start +2 months, ...
dates = pd.DatetimeIndex([start + pd.DateOffset(months=i) for i in range(nt)])


fig, ax = plt.subplots(figsize=(8,4))
ax.plot(dates, dofs_time, marker='o', linestyle='-')

ax.set_xlabel('time')
ax.set_ylabel(r'DOFS$_{\mathrm{state}}$ (trace$(I - P_p\,P_a^{-1})$)')
ax.set_title('DOFS (state) time series')
ax.grid(True)

# date formatting: show Year-Month, rotate for readability
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig(OUTPUT + '/fig1_dofs_time_series.png', dpi=150)
plt.close()
print("Fig1 saved: fig1_dofs_time_series.png")

# ---------- Diagnostic 2: Fractional Variance Reduction (diag-based) map for a chosen time (use last) ----------

# -------------------------------
# Load region masks
# -------------------------------
import cartopy.crs as ccrs
import cartopy.feature as cfeature
region_file = '/scratch/local/for_gc_test/enkf_oco2_inv_v14/rerun_v14/surface_flux/reg_flux_477_ml.2x2.5.nc'
ds_region = xr.open_dataset(region_file)
map_da = ds_region['map']  # dims: (lon, lat, layer)

# map_da: shape (lon, lat, layer)
lon_vals = map_da.longitude.values
lat_vals = map_da.latitude.values

years = np.unique(dates.year)
err0 = np.ma.filled(err0, fill_value=0.0)
err  = np.ma.filled(err,  fill_value=0.0)
for yr in years:
    # Select indices for this year
    idx = np.where(dates.year == yr)[0]
    if len(idx) == 0:
        continue

    # Compute FVR per month
    fvr_list = []
    for t_choice in idx:
        Pa = err0[t_choice,:,:]
        Pp = err[t_choice,:,:]
        # safest diagonal extraction
        if Pa.shape[0]==Pa.shape[1]:
            diag_pa = np.diag(Pa)
            diag_pp = np.diag(Pp)
        else:
            m = min(Pa.shape)
            diag_pa = np.diag(Pa[:m,:m])
            diag_pp = np.diag(Pp[:m,:m])
        with np.errstate(divide='ignore', invalid='ignore'):
            fvr_diag = 1.0 - np.nan_to_num(diag_pp/diag_pa, nan=0.0, posinf=0.0, neginf=0.0)
        fvr_list.append(fvr_diag)

    # Annual mean
    fvr_annual = np.mean(np.stack(fvr_list, axis=0), axis=0)  # (layer,)

    # Map to lon-lat using the generic function
    fvr_map = apply_region_mapping(map_da, fvr_annual)  # now returns plain numpy array

    # --------- Plot Map ---------
    fig = plt.figure(figsize=(12,5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    im = ax.pcolormesh(lon_vals, lat_vals, fvr_map.T,  # transpose so lat=rows
                       cmap='viridis', vmin=0, vmax=1)

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
    ax.set_global()
    ax.set_title(f'Annual Average Fractional Variance Reduction ({yr})', fontsize=14)

    # Colorbar with proper size and label
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7, pad=0.02)
    cbar.set_label('Fractional Variance Reduction', fontsize=12)

    fname = OUTPUT + f'/fig2_fvr_map_{yr}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Fig2 saved: {fname}")

# ---------- Diagnostic 3: Transform-based variance reduction (requires xtm & anomalies) ----------
# We do not have X_prior anomalies in files; attempt to build approximate anomalies from Pa:
# If Pa square and positive-semidef, we can do eig-decomp Pa = U S U^T and set X_prior = U sqrt(S)
# Then we can (only if xtm has compatible dimension) compute post-variances via X_post = X_prior @ T
for yr in years:
    idx = np.where(dates.year == yr)[0]
    if len(idx) == 0:
        continue

    transform_list = []

    for t_choice in idx:
        Pa_local = err0[t_choice, :, :]
        nstate = Pa_local.shape[0]

        if not xtm_dict:
            continue
        first_key = list(xtm_dict.keys())[0]
        T = xtm_dict[first_key]

        if T is None:
            continue

        # Only proceed if T is square and matches state size
        if T.ndim == 2 and T.shape[0] == T.shape[1] and T.shape[0] == nstate:
            # Approximate sqrt(Pa) using Cholesky
            eps = 1e-12
            try:
                L = nlg.cholesky(Pa_local + np.eye(nstate)*eps)
            except nlg.LinAlgError:
                # fallback: eigen-decomp
                eigs, U = nlg.eigh(Pa_local)
                eigs[eigs < 0] = 0.0
                L = U @ np.sqrt(np.diag(eigs))

            X_prior = L
            X_post  = X_prior @ T

            prior_var = np.var(X_prior, axis=1)
            post_var  = np.var(X_post, axis=1)

            with np.errstate(divide='ignore', invalid='ignore'):
                transform_fraction = 1.0 - np.nan_to_num(post_var/prior_var,
                                                         nan=0.0, posinf=0.0, neginf=0.0)
            transform_list.append(transform_fraction)
        else:
            print(f"xtm size {T.shape} does not match nstate={nstate}. Skipping step {t_choice}.")

    if len(transform_list) == 0:
        print(f"No valid transform-based FVR for year {yr}. Skipping.")
        continue

    # Annual mean
    transform_annual = np.mean(np.stack(transform_list, axis=0), axis=0)  # (layer,)

    # Map to lon-lat
    transform_annual_map = apply_region_mapping(map_da, transform_annual)  # plain numpy array

    # --------- Plot Map ---------
    fig = plt.figure(figsize=(12,5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    im = ax.pcolormesh(lon_vals, lat_vals, transform_annual_map.T,  # transpose: lat=rows
                       cmap='viridis', vmin=0, vmax=1)

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
    ax.set_global()
    ax.set_title(f'Annual Average Transform-based VR ({yr})', fontsize=14)

    cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7, pad=0.02)
    cbar.set_label('Transform-based Variance Reduction', fontsize=12)

    fname = OUTPUT + f'/fig3_transform_fvr_map_{yr}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Fig3 saved: {fname}")


# # ---------- Diagnostic 4: Averaging-kernel diagonal map AK_diag ----------
# # AK_diag = 1 - diag(Pp @ pinv(Pa))
# if Pa.shape[0] == Pa.shape[1]:
#     Pa_inv = safe_pinv(Pa)
#     AK_mat = np.eye(Pa.shape[0]) - Pp @ Pa_inv
#     AK_diag = np.diag(AK_mat)
# else:
#     AK_diag = fvr_diag  # fallback
# plt.figure(figsize=(10,4))
# plt.plot(np.arange(AK_diag.size), AK_diag)
# plt.xlabel('state index')
# plt.ylabel('Averaging kernel diagonal (AK_diag)')
# plt.title(f'Fig4: Averaging kernel diagonal at t={t_choice}')
# plt.grid(True)
# plt.savefig('fig4_ak_diag_t{}.png'.format(t_choice), dpi=150)
# plt.close()
# print("Fig4 saved: fig4_ak_diag_t{}.png".format(t_choice))

# ---------- Diagnostic 5: Eigenvalue spectra prior & posterior ----------
# compute eigenvalues (descending) for Pa and Pp (if square)
for yr in years:
    idx = np.where(dates.year == yr)[0]
    if len(idx) == 0:
        continue

    # Annual mean Pa and Pp (consistent with your FVR diagnostics)
    Pa_annual = np.mean(err0[idx, :, :], axis=0)
    Pp_annual = np.mean(err[idx, :, :], axis=0)

    # Only proceed if square
    if Pa_annual.shape[0] == Pa_annual.shape[1]:
        eigs_pa = eigvalsh(Pa_annual)
        eigs_pp = eigvalsh(Pp_annual)

        # Sort descending
        eigs_pa = eigs_pa[::-1]
        eigs_pp = eigs_pp[::-1]

        # Participation ratios
        pr_pa = participation_ratio(eigs_pa)
        pr_pp = participation_ratio(eigs_pp)

        plt.figure(figsize=(6,4))
        plt.semilogy(np.arange(1, len(eigs_pa)+1), eigs_pa, marker='o', label='prior')
        plt.semilogy(np.arange(1, len(eigs_pp)+1), eigs_pp, marker='o', label='posterior')
        plt.xlabel('mode index (descending eigenvalues)')
        plt.ylabel('eigenvalue (log scale)')
        plt.title(f'Fig5: eigenvalue spectra ({yr})\nPR_prior={pr_pa:.2f}, PR_post={pr_pp:.2f}')
        plt.legend()
        plt.grid(True)

        fname = OUTPUT + f'/fig5_eig_spectra_{yr}.png'
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Fig5 saved: {fname} (PR_prior={pr_pa:.2f}, PR_post={pr_pp:.2f})")
    else:
        print(f"Skipping eig spectrum for year {yr}: Pa not square.")

# ---------- Diagnostic 5*: Compare DOFS from fig 1 vs PR timeseries ----------
dofs_time = np.full(nt, np.nan)
pr_prior_time = np.full(nt, np.nan)
pr_post_time  = np.full(nt, np.nan)

for t in range(nt):
    Pa = err0[t,:,:]
    Pp = err[t,:,:]

    # ensure symmetric
    Pa = 0.5*(Pa + Pa.T)
    Pp = 0.5*(Pp + Pp.T)

    # Compute DOFS
    if Pa.shape[0] == Pa.shape[1]:
        Pa_inv = np.linalg.pinv(Pa)
        dofs_time[t] = np.trace(np.eye(Pa.shape[0]) - Pp @ Pa_inv)

        # Participation ratio
        eigs_pa = eigvalsh(Pa)
        eigs_pp = eigvalsh(Pp)
        pr_prior_time[t] = participation_ratio(eigs_pa)
        pr_post_time[t]  = participation_ratio(eigs_pp)
    else:
        # fallback to diag-based approximation
        diag_pa = np.diag(Pa[:min(Pa.shape)])
        diag_pp = np.diag(Pp[:min(Pp.shape)])
        fvr = 1.0 - np.nan_to_num(diag_pp/diag_pa, nan=0.0, posinf=0.0, neginf=0.0)
        dofs_time[t] = np.sum(fvr)
        pr_prior_time[t] = np.sum(diag_pa>0)  # rough proxy
        pr_post_time[t]  = np.sum(diag_pp>0)  # rough proxy

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(dates, dofs_time, marker='o', linestyle='-', label='DOFS (posterior)')
ax.plot(dates, pr_prior_time, marker='x', linestyle='--', label='PR_prior')
ax.plot(dates, pr_post_time, marker='s', linestyle='-.', label='PR_post')

ax.set_xlabel('Time')
ax.set_ylabel('Number of modes / DOF')
ax.set_title('Monthly DOFS and Participation Ratios')
ax.grid(True)
ax.legend()

# Format dates
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig(OUTPUT + '/fig5_dofs_vs_pr_monthly.png', dpi=150)
plt.close()
print("Fig5 saved: fig5_dofs_vs_pr_monthly.png")
# ---------- Diagnostic 5*: Compare DOFS from fig 1 vs PR map plots----------
seasons = {
    'DJF': [12, 1, 2],
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11]
}


for yr in years:
    idx_year = np.where(dates.year == yr)[0]
    if len(idx_year) == 0:
        continue

    for season_name, months in seasons.items():
        idx_season = [i for i in idx_year if dates[i].month in months]
        if len(idx_season) == 0:
            continue

        # Compute seasonal average DOFS matrix
        dofs_matrices = []
        pr_prior_list = []
        pr_post_list = []

        for t in idx_season:
            Pa = err0[t,:,:]
            Pp = err[t,:,:]

            # DOFS matrix
            Pa_inv = np.linalg.pinv(Pa)
            AK = np.eye(Pa.shape[0]) - Pp @ Pa_inv
            dofs_matrices.append(AK)

            # Participation ratio vectors
            eigs_pa = np.linalg.eigvalsh(Pa)
            eigs_pp = np.linalg.eigvalsh(Pp)
            pr_prior_list.append(participation_ratio(eigs_pa))
            pr_post_list.append(participation_ratio(eigs_pp))

        # Seasonal average
        dofs_season_avg = np.mean(np.stack(dofs_matrices, axis=0), axis=0)
        pr_prior_season_avg = np.mean(pr_prior_list, axis=0)  # already scalar here
        pr_post_season_avg  = np.mean(pr_post_list, axis=0)

        # Map DOFS and PR to lat-lon
        dofs_map = apply_region_mapping_matrix(map_da, dofs_season_avg)
        # pr_prior_map = apply_region_mapping(map_da, pr_prior_season_avg)
        # pr_post_map  = apply_region_mapping(map_da, pr_post_season_avg)

        # Diff maps
        diff_prior = dofs_map - pr_prior_season_avg
        diff_post  = dofs_map - pr_post_season_avg
       

        for var_map, title, fname in zip(
            [dofs_map, diff_prior, diff_post],
            ['DOFS', 'DOFS - PR_prior', 'DOFS - PR_post'],
            [f'dofs_{yr}_{season_name}.png',
            f'diff_prior_{yr}_{season_name}.png',
            f'diff_post_{yr}_{season_name}.png']
        ):

            fig = plt.figure(figsize=(12,5))
            ax = plt.axes(projection=ccrs.PlateCarree())

            if "DOFS -" in title:
                im = ax.pcolormesh(map_da.longitude.values,
                                map_da.latitude.values,
                                var_map.T,  # keep original orientation
                                cmap='Reds_r',
                                
                                )
            else:
                im = ax.pcolormesh(map_da.longitude.values,
                                map_da.latitude.values,
                                var_map.T,
                                cmap='Reds',)

            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.set_global()
            ax.set_title(f'{title} ({season_name} {yr})', fontsize=14)

            cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7, pad=0.02)
            cbar.set_label(title, fontsize=12)
            plt.savefig(f'{OUTPUT}/fig5_{fname}', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {fname}")

seasons = {
    'DJF': [12, 1, 2],
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11]
}

for yr in years:
    idx_year = np.where(dates.year == yr)[0]
    if len(idx_year) == 0:
        continue

    for season_name, months in seasons.items():
        idx_season = [i for i in idx_year if dates[i].month in months]
        if len(idx_season) == 0:
            continue

        # ---------- compute seasonal average uncertainty reduction ----------
        fvr_list = []
        for t in idx_season:
            Pa = err0[t,:,:]  # prior covariance (nregions x nregions)
            Pp = err[t,:,:]   # posterior covariance

            # fractional variance reduction (matrix form)
            # FVR_mat = 100 * (Pa - Pp)/Pa  # elementwise reduction in cov
            # fvr_list.append(FVR_mat)
            # print(np.nanmax(FVR_mat), np.nanmin(FVR_mat))
            FVR_mat = np.zeros_like(Pa)
            mask = Pa != 0
            FVR_mat[mask] = 100 * (Pa[mask] - Pp[mask]) / Pa[mask]
            print(np.nanmax(FVR_mat), np.nanmin(FVR_mat))
            fvr_list.append(FVR_mat)
        # seasonal mean reduction
        FVR_season_avg = np.mean(np.stack(fvr_list, axis=0), axis=0)  # still nregions x nregions
        print(FVR_season_avg)
        # Map to lon-lat grid
        fvr_map_percent = apply_region_mapping_matrix(map_da, FVR_season_avg)  # shape (lon, lat)
        # print(fvr_map_percent.min(), fvr_map_percent.max(), FVR_season_avg.min(), FVR_season_avg.max(), fvr_map_percent.shape)
        # print(fvr_map_percent)
        # ---------- plot ----------
        fig = plt.figure(figsize=(12,5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        im = ax.pcolormesh(map_da.longitude.values,
                           map_da.latitude.values,
                           fvr_map_percent.T,  # lat as rows
                           cmap='bwr', vmin=-100, vmax=100)
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
        ax.set_global()
        ax.set_title(f'Seasonal Uncertainty Reduction (%) ({season_name} {yr})', fontsize=14)

        cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7, pad=0.02)
        cbar.set_label('Reduction (%)', fontsize=12)

        plt.tight_layout()
        fname = f'{OUTPUT}/Error_reduction_{yr}_{season_name}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fname}")

seasons = {
    'DJF': [12, 1, 2],
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11]
}

eps = 0  # threshold to avoid dividing by tiny flux values

for yr in years:
    idx_year = np.where(dates.year == yr)[0]
    if len(idx_year) == 0:
        continue

    for season_name, months in seasons.items():
        idx_season = [i for i in idx_year if dates[i].month in months]
        if len(idx_season) == 0:
            continue

        # ---------- accumulate seasonal reductions ----------
        abs_list = []
        frac_list = []

        for t in idx_season:
            Pa = err0[t, :, :]  # prior covariance (nregions x nregions)
            Pp = err[t, :, :]   # posterior covariance

            # --- absolute reduction (flux units) ---
            
            # --- map Pa/Pp to lon-lat ---
            Pa_map = apply_region_mapping_matrix(map_da, Pa)
            Pp_map = apply_region_mapping_matrix(map_da, Pp)
            abs_red = Pa_map - Pp_map
            abs_list.append(abs_red)


            # Assume flux_map is the actual flux on lon-lat grid at this timestep
            flux_map = apply_region_mapping(map_da, flux[t, :])  

            # --- fractional reduction ---
            mask = np.abs(flux_map) > eps
            frac_red = np.zeros_like(flux_map)
            frac_red[mask] = 100.0 * (Pa_map[mask] - Pp_map[mask]) / (flux_map[mask]**2)
            frac_list.append(frac_red)

        # seasonal mean
        abs_season_avg = np.mean(np.stack(abs_list, axis=0), axis=0)
        frac_season_avg = np.mean(np.stack(frac_list, axis=0), axis=0)

        # ---------- plot absolute reduction ----------
        fig = plt.figure(figsize=(12, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        im = ax.pcolormesh(map_da.longitude.values,
                           map_da.latitude.values,
                           np.sqrt(abs_season_avg.T),
                           cmap='viridis')
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
        ax.set_global()
        ax.set_title(f'Seasonal Absolute Uncertainty Reduction (PgC/yr) ({season_name} {yr})', fontsize=14)
        plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7, pad=0.02, label='PgC/yr')
        plt.tight_layout()
        fname = f'{OUTPUT}/Absolute_Error_reduction_{yr}_{season_name}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fname}")

        # ---------- plot fractional reduction ----------
        fig = plt.figure(figsize=(12, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        im = ax.pcolormesh(map_da.longitude.values,
                           map_da.latitude.values,
                           np.sqrt(frac_season_avg.T),
                           cmap='bwr', vmin=-100, vmax=100)
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
        ax.set_global()
        ax.set_title(f'Seasonal Fractional Uncertainty Reduction (%) ({season_name} {yr})', fontsize=14)
        plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7, pad=0.02, label='Reduction (%)')
        plt.tight_layout()
        fname = f'{OUTPUT}/Fractional_Error_reduction_{yr}_{season_name}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fname}")


from concurrent.futures import ThreadPoolExecutor
def compute_one_season(yr, season_name, idx_season, err0, err, flux, map_da, eps=0):
    abs_list, frac_list = [], []

    for t in idx_season:
        Pa = err0[t, :, :]
        Pp = err[t, :, :]

        # map Pa/Pp to lon-lat
        Pa_map = apply_region_mapping_matrix(map_da, Pa)
        Pp_map = apply_region_mapping_matrix(map_da, Pp)

        # absolute reduction
        abs_list.append(Pa_map - Pp_map)

        # fractional reduction
        flux_map = apply_region_mapping(map_da, flux[t, :])
        mask = np.abs(flux_map) > eps
        frac_red = np.zeros_like(flux_map)
        frac_red[mask] = 100.0 * (Pa_map[mask] - Pp_map[mask]) / (flux_map[mask]**2)
        frac_list.append(frac_red)

    abs_season_avg = np.mean(np.stack(abs_list, axis=0), axis=0)
    frac_season_avg = np.mean(np.stack(frac_list, axis=0), axis=0)

    return yr, season_name, abs_season_avg, frac_season_avg

# ------------------------------
# Function to compute all seasonal reductions in parallel
# ------------------------------
def compute_seasonal_reductions_parallel(years, dates, err0, err, flux, map_da, eps=0, max_workers=8):
    seasons = {
        'DJF': [12, 1, 2],
        'MAM': [3, 4, 5],
        'JJA': [6, 7, 8],
        'SON': [9, 10, 11]
    }

    tasks = []
    for yr in years:
        idx_year = np.where(dates.year == yr)[0]
        if len(idx_year) == 0:
            continue
        for season_name, months in seasons.items():
            idx_season = [i for i in idx_year if dates[i].month in months]
            if len(idx_season) == 0:
                continue
            tasks.append((yr, season_name, idx_season, err0, err, flux, map_da, eps))

    reductions = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compute_one_season, *task): task[:2] for task in tasks}
        for fut in as_completed(futures):
            yr, season_name, abs_avg, frac_avg = fut.result()
            reductions[(yr, season_name)] = {'abs': abs_avg, 'frac': frac_avg}

    return reductions



def plot_one(key, data, output_dir, abs_vmin, abs_vmax, title_s='Seasonal'):
    yr, season_name = key      # <-- correct unpacking

    abs_season_avg = data['abs']
    frac_season_avg = data['frac']

    # --- absolute reduction ---
    fig = plt.figure(figsize=(12, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    im = ax.pcolormesh(
        map_da.longitude.values,
        map_da.latitude.values,
        np.sqrt(abs_season_avg.T),
        cmap='viridis',
        vmin=abs_vmin,
        vmax=abs_vmax
    )
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
    ax.set_global()
    ax.set_title(f'{title_s} Absolute Uncertainty Reduction (PgC/yr) ({season_name} {yr})', fontsize=14)
    plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7, pad=0.02, label='PgC/yr')
    plt.tight_layout()
    fname = f'{output_dir}/Absolute_Error_reduction_{yr}_{season_name}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")

    # --- fractional reduction ---
    fig = plt.figure(figsize=(12, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    im = ax.pcolormesh(
        map_da.longitude.values,
        map_da.latitude.values,
        np.sqrt(frac_season_avg.T),
        cmap='bwr',
        vmin=-100,
        vmax=100
    )
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
    ax.set_global()
    ax.set_title(f'{title_s} Fractional Uncertainty Reduction (%) ({season_name} {yr})', fontsize=14)
    plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7, pad=0.02, label='Reduction (%)')
    plt.tight_layout()
    fname = f'{output_dir}/Fractional_Error_reduction_{yr}_{season_name}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


# ------------------------------
# Function to plot all reductions in parallel
# ------------------------------
from concurrent.futures import as_completed
def plot_seasonal_reductions_parallel(reductions, map_da, output_dir, abs_vmin=None, abs_vmax=None, max_workers=4):


    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(plot_one, key, data, output_dir, abs_vmin, abs_vmax)
            for key, data in reductions.items()
        ]
        for fut in as_completed(futures):
            fut.result()  # just wait for completion

# ------------------------------
# Usage
# ------------------------------
reductions = compute_seasonal_reductions_parallel(years, dates, err0, err, flux, map_da, eps=0, max_workers=1)

# unified colorbar for absolute reductions
all_abs = np.stack([data['abs'] for data in reductions.values()], axis=0)
abs_vmin = np.sqrt(all_abs.min())
abs_vmax = np.sqrt(all_abs.max())

plot_seasonal_reductions_parallel(reductions, map_da, OUTPUT, abs_vmin=abs_vmin, abs_vmax=abs_vmax, max_workers=1)

yearly_reductions = {}

for yr in years:
    # collect all seasons for this year
    season_keys = [(yr, s) for s in ['DJF', 'MAM', 'JJA', 'SON'] if (yr, s) in reductions]

    if len(season_keys) == 0:
        continue

    abs_maps = [reductions[key]['abs'] for key in season_keys]
    frac_maps = [reductions[key]['frac'] for key in season_keys]

    yearly_abs = np.mean(np.stack(abs_maps, axis=0), axis=0)
    yearly_frac = np.mean(np.stack(frac_maps, axis=0), axis=0)

    yearly_reductions[yr] = {"abs": yearly_abs, "frac": yearly_frac}

def plot_yearly_reductions(yearly_reductions, map_da, output_dir, abs_vmin, abs_vmax, max_workers=4):

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                plot_one,
                (yr, "ANNUAL"),  # key format reused
                data,
                output_dir,
                abs_vmin,
                abs_vmax,
                title_s = Annual
            )
            for yr, data in yearly_reductions.items()
        ]

        for fut in as_completed(futures):
            fut.result()


plot_yearly_reductions(
    yearly_reductions,
    map_da,
    OUTPUT,
    abs_vmin,
    abs_vmax,
    max_workers=1
)



reductions = compute_seasonal_reductions_parallel(years, dates, err0, err, flux, map_da, eps=0, max_workers=8)

# Determine absolute color range
all_abs = np.stack([d['abs'] for d in reductions.values()], axis=0)
abs_vmin = np.sqrt(all_abs.min())
abs_vmax = np.sqrt(all_abs.max())

# Plot seasonal maps
plot_seasonal_reductions_parallel(reductions, map_da, OUTPUT, abs_vmin=abs_vmin, abs_vmax=abs_vmax, max_workers=4)

# Aggregate yearly reductions
yearly_reductions = {}
for yr in years:
    keys = [(yr, s) for s in ['DJF','MAM','JJA','SON'] if (yr,s) in reductions]
    if len(keys) == 0:
        continue
    abs_maps = [reductions[k]['abs'] for k in keys]
    frac_maps = [reductions[k]['frac'] for k in keys]
    yearly_reductions[yr] = {
        'abs': np.mean(np.stack(abs_maps, axis=0), axis=0),
        'frac': np.mean(np.stack(frac_maps, axis=0), axis=0)
    }

# Plot yearly maps
plot_yearly_reductions(yearly_reductions, map_da, OUTPUT, abs_vmin, abs_vmax, max_workers=4)