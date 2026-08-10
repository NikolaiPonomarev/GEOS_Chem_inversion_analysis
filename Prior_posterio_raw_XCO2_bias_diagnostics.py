import os
import glob
import re
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import TwoSlopeNorm

norm_RB = TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=5.0)

def diagnose_assimilation_bias_maps(
    pattern="/scratch/local/nponomar/african_inversion_full_setup/gc_enkf_inversion_africa/oco_inv/obs_oco_assim_step.*.nc",
    outdir="./assim_diagnostic_maps",
    quantiles=[0.05, 0.25, 0.50, 0.75, 0.95],
    eps=1e-6,
    bias_vmin=-3,
    bias_vmax=3,
    coord_decimals=3   # round lon/lat to N decimals when grouping unique locations
):
    """
    1) Prints daily assimilation-ratio quantiles (text only).
    2) Saves daily PNGs: [Prior Bias map | Posterior Bias map].
    3) Saves final composite PNG: [Mean Prior Bias | Mean Posterior Bias]
       averaged per unique (lon, lat) coordinate across all days.
    """
    os.makedirs(outdir, exist_ok=True)
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files found matching: {pattern}")
        return

    # Accumulators for the composite (all-days, per unique location)
    all_lons, all_lats = [], []
    all_prior_bias, all_post_bias = [], []
    daily_dates, daily_obs_mean, daily_prior_mean, daily_post_mean = [], [], [], []

    # --- Print header ---
    q_labels = " ".join([f"q{int(q*100):02d}" for q in quantiles])
    header = f"{'Date':<12} {'N_obs':>8} {'Median':>8} {q_labels}"
    print(header)
    print("-" * len(header))

    for fpath in files:
        # Extract YYYYMMDD
        m = re.search(r'obs_oco_assim_step\.(\d{8})\.nc', fpath)
        if not m:
            continue
        date_str = m.group(1)

        try:
            ds = xr.open_dataset(fpath)

            # Load and flatten
            obs   = ds['obs'].values.ravel()
            prior = ds['mod'].values.ravel()
            mod_adj      = ds['mod_adj'].values.ravel()
            new_mod_adj  = ds['new_mod_adj'].values.ravel()
            lon = ds['lon'].values.ravel()
            lat = ds['lat'].values.ravel()

            # Posterior as defined by the user
            posterior = prior + mod_adj + new_mod_adj

            # Valid mask
            valid = (np.isfinite(obs) & np.isfinite(prior) & np.isfinite(posterior) &
                     np.isfinite(lon) & np.isfinite(lat))
            obs, prior, posterior, lon, lat = [
                a[valid] for a in [obs, prior, posterior, lon, lat]
            ]
            n_obs = obs.size
            if n_obs == 0:
                continue

            # --- Ratio quantiles (printed only) ---
            numerator   = np.abs(prior - obs)
            denominator = np.maximum(np.abs(posterior - obs), eps)
            ratio = numerator / denominator
            # ratio = np.clip(ratio, 0.01, 100.0)

            q_values = np.nanquantile(ratio, quantiles)
            median = q_values[quantiles.index(0.50)] if 0.50 in quantiles else np.nan
            q_str = " ".join([f"{q:8.3f}" for q in q_values])
            print(f"{date_str:<12} {n_obs:>8} {median:>8.3f} {q_str}")

            # --- Biases ---
            prior_bias = prior - obs
            post_bias  = posterior - obs

            # --- Accumulate daily global means for timeseries ---
            daily_obs_mean.append(np.mean(obs))
            daily_prior_mean.append(np.mean(prior))
            daily_post_mean.append(np.mean(posterior))
            daily_dates.append(pd.to_datetime(date_str, format='%Y%m%d'))

            # Accumulate for composite
            all_lons.append(lon)
            all_lats.append(lat)
            all_prior_bias.append(prior_bias)
            all_post_bias.append(post_bias)

            # ============================
            # Daily figure: Prior | Posterior bias
            # ============================
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Prior bias
            ax = axes[0]
            sc = ax.scatter(lon, lat, c=prior_bias, cmap='RdBu_r',
                            vmin=bias_vmin, vmax=bias_vmax, s=20,
                            alpha=0.85, edgecolors='none')
            mp_all = np.mean(prior_bias)
            ax.set_title(f'Prior Bias   {date_str}, (mean={mp_all:+.2f} ppm)')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_xlim(lon.min()-5, lon.max()+5)
            ax.set_ylim(lat.min()-5, lat.max()+5)
            cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
            cbar.set_label('Prior - Obs (ppm)')

            # Posterior bias
            ax = axes[1]
            sc = ax.scatter(lon, lat, c=post_bias, cmap='RdBu_r',
                            vmin=bias_vmin, vmax=bias_vmax, s=20,
                            alpha=0.85, edgecolors='none')
            mp_all = np.mean(post_bias)
            ax.set_title(f'Posterior Bias   {date_str}, (mean={mp_all:+.2f} ppm)')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_xlim(lon.min()-5, lon.max()+5)
            ax.set_ylim(lat.min()-5, lat.max()+5)
            ax.xaxis.set_major_locator(MultipleLocator(10))  
            ax.yaxis.set_major_locator(MultipleLocator(10))   # grid every 10° lat
            ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6, color='gray')
            cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
            cbar.set_label('Posterior - Obs (ppm)')

            plt.tight_layout()
            fig.savefig(os.path.join(outdir, f"bias_map_{date_str}.png"),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

            ds.close()

        except Exception as e:
            print(f"{date_str:<12} ERROR: {e}")
            continue

    # ============================
    # Composite figure: average per unique (lon, lat)
    # ============================
    if not all_lons:
        print("No valid data for composite.")
        return

    # Concatenate everything
    all_lon = np.concatenate(all_lons)
    all_lat = np.concatenate(all_lats)
    all_pb  = np.concatenate(all_prior_bias)
    all_pob = np.concatenate(all_post_bias)

    # Group by rounded unique coordinates
    lon_r = np.round(all_lon, coord_decimals)
    lat_r = np.round(all_lat, coord_decimals)
    coords = np.column_stack([lon_r, lat_r])

    unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)
    n_unique = unique_coords.shape[0]

    # Compute mean bias per unique location
    counts = np.bincount(inverse, minlength=n_unique).astype(float)
    mean_prior_bias = np.bincount(inverse, weights=all_pb,  minlength=n_unique) / counts
    mean_post_bias  = np.bincount(inverse, weights=all_pob, minlength=n_unique) / counts

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Mean prior bias (per unique location)
    ax = axes[0]
    mp_all = np.mean(mean_prior_bias)
    sc = ax.scatter(unique_coords[:, 0], unique_coords[:, 1],
                    c=mean_prior_bias, cmap='RdBu_r',
                    vmin=bias_vmin, vmax=bias_vmax, s=20,
                    alpha=0.85, edgecolors='none')
    ax.set_title(f'Mean Prior Bias\n({n_unique} unique locations, all days),\n (mean={mp_all:+.2f} ppm)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xlim(all_lon.min()-5, all_lon.max()+5)
    ax.set_ylim(all_lat.min()-5, all_lat.max()+5)
    ax.xaxis.set_major_locator(MultipleLocator(10))   
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6, color='gray')
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label('Mean Prior - Obs (ppm)')

    # Mean posterior bias (per unique location)
    ax = axes[1]
    sc = ax.scatter(unique_coords[:, 0], unique_coords[:, 1],
                    c=mean_post_bias, cmap='RdBu_r',
                    vmin=bias_vmin, vmax=bias_vmax, s=20,
                    alpha=0.85, edgecolors='none')
    mp_all = np.mean(mean_post_bias)
    ax.set_title(f'Mean Posterior Bias\n({n_unique} unique locations, all days),\n (mean={mp_all:+.2f} ppm)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xlim(all_lon.min()-5, all_lon.max()+5)
    ax.set_ylim(all_lat.min()-5, all_lat.max()+5)
    ax.xaxis.set_major_locator(MultipleLocator(10))  
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6, color='gray')
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label('Mean Posterior - Obs (ppm)')

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "bias_map_average_unique_locations.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\nSaved {len(files)} daily maps + composite to: {outdir}/")
    print(f"Composite averaged over {n_unique} unique (lon, lat) locations.")

    #plot ratios
    ratio_from_means = np.abs(mean_prior_bias) / np.maximum(np.abs(mean_post_bias), eps)

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Mean prior bias (re-plotted for side-by-side context)
    ax = axes2[0]
    mp_all = np.mean(mean_prior_bias)
    sc = ax.scatter(unique_coords[:, 0], unique_coords[:, 1],
                    c=mean_prior_bias, cmap='RdBu_r',
                    vmin=bias_vmin, vmax=bias_vmax, s=20,
                    alpha=0.85, edgecolors='none')
    ax.set_title(f'Mean Prior Bias   (mean={mp_all:+.2f} ppm)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xlim(unique_coords[:, 0].min()-5, unique_coords[:, 0].max()+5)
    ax.set_ylim(unique_coords[:, 1].min()-5, unique_coords[:, 1].max()+5)
    ax.xaxis.set_major_locator(MultipleLocator(10))  
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6, color='gray')
    plt.colorbar(sc, ax=ax, shrink=0.7, label='Mean Prior - Obs (ppm)')

    # Panel 2: Mean posterior bias
    ax = axes2[1]
    mpo_all = np.mean(mean_post_bias)
    sc = ax.scatter(unique_coords[:, 0], unique_coords[:, 1],
                    c=mean_post_bias, cmap='RdBu_r',
                    vmin=bias_vmin, vmax=bias_vmax, s=20,
                    alpha=0.85, edgecolors='none')
    ax.set_title(f'Mean Posterior Bias   (mean={mpo_all:+.2f} ppm)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xlim(unique_coords[:, 0].min()-5, unique_coords[:, 0].max()+5)
    ax.set_ylim(unique_coords[:, 1].min()-5, unique_coords[:, 1].max()+5)
    ax.xaxis.set_major_locator(MultipleLocator(10))  
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6, color='gray')
    plt.colorbar(sc, ax=ax, shrink=0.7, label='Mean Posterior - Obs (ppm)')

    # Panel 3: Ratio from the averaged biases
    ax = axes2[2]
    mr_all = np.mean(ratio_from_means)
    sc = ax.scatter(unique_coords[:, 0], unique_coords[:, 1],
                    c=ratio_from_means, cmap='RdBu_r', norm=norm_RB, s=20,
                    alpha=0.85, edgecolors='none')
    ax.set_title(f'Ratio from Averaged Bias   (mean={mr_all:.2f})')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xlim(unique_coords[:, 0].min()-5, unique_coords[:, 0].max()+5)
    ax.set_ylim(unique_coords[:, 1].min()-5, unique_coords[:, 1].max()+5)
    ax.xaxis.set_major_locator(MultipleLocator(10))  
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6, color='gray')
    plt.colorbar(sc, ax=ax, shrink=0.7, label='|Mean Prior Bias| / |Mean Post Bias|')

    plt.tight_layout()
    fig2.savefig(os.path.join(outdir, "ratio_from_averaged_bias_maps.png"),
                 dpi=150, bbox_inches='tight')
    plt.close(fig2)
    # ============================
    # Timeseries: global mean XCO2
    # ============================
    if daily_dates:
        fig_ts, ax_ts = plt.subplots(figsize=(10, 5))

        ax_ts.plot(daily_dates, daily_obs_mean,   'o-', color='black',  label='Observations', markersize=5)
        ax_ts.plot(daily_dates, daily_prior_mean, 's-', color='tab:blue', label='Prior', markersize=5)
        ax_ts.plot(daily_dates, daily_post_mean,  '^-', color='tab:red',  label='Posterior', markersize=5)

        ax_ts.set_xlabel('Date')
        ax_ts.set_ylabel('Domain Mean XCO2 (ppm)')
        ax_ts.set_title('Daily Domain Mean XCO2')
        ax_ts.legend(loc='best')
        ax_ts.grid(True, linestyle='--', alpha=0.5)
        fig_ts.autofmt_xdate()

        plt.tight_layout()
        fig_ts.savefig(os.path.join(outdir, "timeseries_domain_mean_xco2.png"),
                       dpi=150, bbox_inches='tight')
        plt.close(fig_ts)
# -------------------------------------------------------------------------
# Run
# -------------------------------------------------------------------------
if __name__ == "__main__":
    diagnose_assimilation_bias_maps()