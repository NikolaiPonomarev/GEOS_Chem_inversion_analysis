import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import cartopy.feature as cfeature
import cartopy.crs as ccrs


def make_diverging_bivariate_cmap(n=256, sat_min=0.40, center_sharpness_b=2.0, center_sharpness_r=0.8):
    """
    2-D colormap:
      X (cols): negative (red) -> 0 (white) -> positive (blue)
      Y (rows): low count (pale) -> high count (saturated)

    Parameters
    ----------
    sat_min : float
        Minimum saturation even at lowest count (0-1). Higher = less pale.
    center_sharpness_r : float
        How abruptly the color goes to white at zero bias.
        1.0 = linear (wide pink band). 4.0 = sharp white only exactly at 0.
    """
    grid = np.zeros((n, n, 3))
    for i in range(n):
        fy = i / (n - 1)
        sat = sat_min + (1.0 - sat_min) * fy   # e.g. 0.40 -> 1.00
        for j in range(n):
            fx = j / (n - 1)

            if fx <= 0.5:
                t = (fx / 0.5) ** center_sharpness_b
                r, g, b = t, t, 1.0

            else:
                t = ((fx - 0.5) / 0.5) ** center_sharpness_r
                r, g, b = 1.0, 1.0 - t, 1.0 - t

            base = np.array([r, g, b])
            white = np.array([1.0, 1.0, 1.0])
            c = white * (1.0 - sat) + base * sat
            grid[i, j] = np.clip(c, 0, 1)
    return grid


def plot_bivariate_gridded(ds, outdir, eps=1e-6,
                           bias_vmin=-3, bias_vmax=3,
                           sat_min=0.40, center_sharpness_b=4.0, center_sharpness_r=0.25):
    """
    Plot bivariate map from ALREADY GRIDDED xarray dataset.
    Uses imshow (fast) instead of scatter (correct for regular grids).
    """
    os.makedirs(outdir, exist_ok=True)

    # --- 1. Time-aggregate ---
    prior_bias = ds['prior'] - ds['obs']          # (time, lat, lon)
    post_bias  = ds['optimized'] - ds['obs']

    mean_prior = np.abs(prior_bias.mean(dim='time'))
    mean_post  = np.abs(post_bias.mean(dim='time'))

    diff = (mean_prior - mean_post).values      # (lat, lon)

    # Sum counts over time
    counts = ds['counts'].sum(dim='time').values  # (lat, lon)

    nlat, nlon = diff.shape
    lon = ds['lon'].values
    lat = ds['lat'].values

    # --- 2. Normalize X: signed bias difference ---
    abs_max = max(abs(float(bias_vmin)), abs(float(bias_vmax)),
                  np.percentile(np.abs(diff[np.isfinite(diff)]), 99.5))
    bmin, bmax = -abs_max, abs_max

    bias_norm = (diff - bmin) / (bmax - bmin + eps)
    bias_norm = np.clip(bias_norm, 0.0, 1.0)
    valid_counts = counts[np.isfinite(counts)]
    cmin = float(valid_counts.min())
    cmax = float(valid_counts.max())
    print(f'Counts range: {cmin:.0f} to {cmax:.0f}')

    count_norm = (counts - cmin) / (cmax - cmin + eps)
    count_norm = np.clip(count_norm, 0.0, 1.0)

    cmap_bv = make_diverging_bivariate_cmap(n=256, sat_min=sat_min, 
                                    center_sharpness_b=center_sharpness_b, center_sharpness_r=center_sharpness_r)

    ix = (bias_norm * 255).astype(int)
    iy = (count_norm * 255).astype(int)

    # Start with white background for invalid/missing cells
    rgb_grid = np.ones((nlat, nlon, 3))
    valid = np.isfinite(diff) & np.isfinite(counts) & (counts > 0)
    rgb_grid[valid] = cmap_bv[iy[valid], ix[valid]]

    # --- 5. Plot with imshow (regular grid = no scatter needed) ---
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})

    # extent covers the full lon/lat range; imshow stretches pixels to fit
    im = ax.imshow(rgb_grid, origin='lower', aspect='auto', extent=[float(lon.min()), float(lon.max()), \
                float(lat.min()), float(lat.max())], transform=ccrs.PlateCarree(), interpolation='nearest')

    ax.set_title('Bias Change vs. Observation Count')
    ax.set_xlim(float(lon.min()) - 5, float(lon.max()) + 5)
    ax.set_ylim(float(lat.min()) - 5, float(lat.max()) + 5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color='gray',
        alpha=0.6,
        linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = MultipleLocator(10)
    gl.ylocator = MultipleLocator(10)
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)

    ax_inset = ax.inset_axes([0.10, 0.13, 0.20, 0.22]) # [x0, y0, width, height] in axes fraction

    ax_inset.imshow(cmap_bv, origin='lower', extent=[0, 1, 0, 1], aspect='auto')

    # Crosshair at center
    ax_inset.axvline(0.5, color='gray', linewidth=0.6, alpha=0.7)
    ax_inset.axhline(0.5, color='gray', linewidth=0.6, alpha=0.7)
    ax_inset.grid(True, which='both', linestyle='-', linewidth=0.3, alpha=0.4, color='gray')

    # X labels: real bias values
    ax_inset.set_xticks([0, 0.5, 1])
    ax_inset.set_xticklabels([f'{bmin:.1f}\n(worse)', '0.0\n(white)', f'{bmax:.1f}\n(better)'], fontsize=10)
    ax_inset.set_xlabel('Bias change (ppm)', fontsize=11, labelpad=3)

    # Y labels: real observation counts
    cmid = (cmin + cmax) / 2
    ax_inset.set_yticks([0, 0.5, 1])
    ax_inset.set_yticklabels([f'{cmin:.0f}', f'{cmid:.0f}', f'{cmax:.0f}'], fontsize=10)
    ax_inset.set_ylabel('Obs Count', fontsize=11, labelpad=3)

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "bivariate_gridded.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved to {outdir}/bivariate_gridded.png")



if __name__ == '__main__':
    file = ('/exports/geos.ed.ac.uk/palmer_group/nponomar/' \
            'GEOS_Chem_inversion_analysis_AF/GEOS_Chem_inversion_analysis/' \
            'aggregated/daily_binned_prior_optimized_obs_2016.nc')
    outdir = './bivariate_output'

    ds = xr.open_dataset(file)
    print(ds)

    plot_bivariate_gridded(ds, outdir, \
        bias_vmin=-3, bias_vmax=3, \
        sat_min=0.40, \
        center_sharpness_b=2.0, \
        center_sharpness_r=0.8)