import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point
import matplotlib.patches as mpatches
import rioxarray
from rasterio.enums import Resampling
import os
import itertools
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ----------------------
# Load daily-binned prior/optimized/obs dataset
# ----------------------
ds = xr.open_dataset('/exports/geos.ed.ac.uk/palmer_group/nponomar/GEOSChem_inversions/daily_binned_prior_optimized_obs.nc')

lat = ds['lat'].values
lon = ds['lon'].values
time = ds['time'].values
n_lat, n_lon = len(lat), len(lon)

# Land mask using natural earth
land_mask = np.zeros((n_lat, n_lon), dtype=bool)
land_shp = shpreader.natural_earth(resolution='110m', category='physical', name='land')
land_geoms = list(shpreader.Reader(land_shp).geometries())

for i in range(n_lat):
    for j in range(n_lon):
        pt = Point(lon[j], lat[i])
        land_mask[i, j] = any(pt.within(geom) for geom in land_geoms)

ocean_mask = ~land_mask

# Tropics mask
def lat_band_mask(lat_min, lat_max):
    mask = ((lat >= lat_min) & (lat <= lat_max))[:, None]
    mask = np.repeat(mask, n_lon, axis=1)
    return mask

tropics_mask = lat_band_mask(-23.5, 23.5)

file='/exports/geos.ed.ac.uk/palmer_group/nponomar/Landuse/CCI4SEN2COR/top3_lc_2x2p5.nc'
ds_lc = xr.open_dataset(file)
# ------------------------------
# Extract top 3 land cover classes
# ------------------------------
top3_classes = ds_lc['top3_landcover'].values  # raw int16 codes

# ------------------------------
# Reclass map: map original LC codes to 1-9
# ------------------------------
reclass_map = {
    10: 6, 20: 6, 30: 6,       # cropland
    40: 5, 100: 5, 110: 5,     # mosaic vegetation
    50: 1, 60: 1,              # broadleaf forest
    70: 2, 80: 2, 90: 2,       # needleleaf / mixed forest
    120: 3,                     # shrubland
    130: 4,                     # grassland
    190: 7,                     # urban
    200: 9, 220: 9, 140: 9, 150: 9, 160: 9, 170: 9, 180: 9,  # barren/snow/other -> Other
    210: 8                      # water/ocean
}

# Apply reclass map to all top3 ranks
top3_mapped = np.vectorize(reclass_map.get)(top3_classes)
# Wrap into xarray DataArray
top3_da_mapped = xr.DataArray(
    top3_mapped,
    dims=['rank', 'lat', 'lon'],
    coords={
        'rank': ds_lc['rank'].values,
        'lat': ds_lc['lat'].values,
        'lon': ds_lc['lon'].values
    },
    name='top3_lc'
)

# ------------------------------
# Define LC classes (1–9) for regions
# ------------------------------
lc_classes = {
    'Broadleaf Forest': 1,
    'Needleleaf/Mixed Forest': 2,
    'Shrubland': 3,
    'Grassland': 4,
    'Mosaic Vegetation': 5,
    'Cropland': 6,
    'Urban': 7,
    'Water/Ocean': 8,
    'Other': 9
}

# ------------------------------
# Build regions using all top 3 ranks
# ------------------------------
regions = {}
for name, code in lc_classes.items():
    mask = (top3_da_mapped == code).any(dim='rank')
    regions[name] = mask.values  # convert to numpy array

# Add ocean and tropical forest masks
regions['Ocean'] = ocean_mask
regions['Broadleaf_Tropics'] = tropics_mask & regions['Broadleaf Forest']

def annual_stats(ds_obs, ds_model, ds_corrected, mask):
    """Compute annual bias, rmse, crmse (from daily values), mean, and std for boxplots of obs, model, corrected."""
    obs_masked = ds_obs.where(mask)
    model_masked = ds_model.where(mask)
    corrected_masked = ds_corrected.where(mask)
    
    years = np.unique(ds_obs['time.year'].values)
    bias_list, rmse_list, crmse_list = [], [], []
    bias_list_corr, rmse_list_corr, crmse_list_corr = [], [], []
    mean_obs_list, mean_model_list, mean_corr_list = [], [], []
    std_obs_list, std_model_list, std_corr_list = [], [], []

    for y in years:
        obs_year = obs_masked.sel(time=ds_obs['time.year']==y)
        model_year = model_masked.sel(time=ds_model['time.year']==y)
        corr_year = corrected_masked.sel(time=ds_corrected['time.year']==y)
        
        obs_flat = obs_year.values.flatten()
        model_flat = model_year.values.flatten()
        corr_flat = corr_year.values.flatten()
        
        # Prior bias calculation (prior - obs)
        valid_prior = ~np.isnan(obs_flat) & ~np.isnan(model_flat)
        if np.sum(valid_prior) < 2:
            bias_list.append(np.nan)
            rmse_list.append(np.nan)
            crmse_list.append(np.nan)
            mean_obs_list.append([])
            mean_model_list.append([])
            mean_corr_list.append([])
            std_obs_list.append(np.nan)
            std_model_list.append(np.nan)
            std_corr_list.append(np.nan)
        else:
            diff_prior = model_flat[valid_prior] - obs_flat[valid_prior]
            bias_list.append(np.mean(diff_prior))
            rmse_list.append(np.sqrt(np.mean(diff_prior**2)))
            crmse_list.append(np.sqrt(np.mean((diff_prior - np.mean(diff_prior))**2)))
            
            # Posterior bias calculation (posterior - obs)
            valid_post = ~np.isnan(obs_flat) & ~np.isnan(corr_flat)
            diff_post = corr_flat[valid_post] - obs_flat[valid_post]
            bias_list_corr.append(np.mean(diff_post))
            rmse_list_corr.append(np.sqrt(np.mean(diff_post**2)))
            crmse_list_corr.append(np.sqrt(np.mean((diff_post - np.mean(diff_post))**2)))

            # store full values for boxplots
            mean_obs_list.append(obs_flat[valid_prior])
            mean_model_list.append(model_flat[valid_prior])
            mean_corr_list.append(corr_flat[valid_post])
            std_obs_list.append(np.std(obs_flat[valid_prior]))
            std_model_list.append(np.std(model_flat[valid_prior]))
            std_corr_list.append(np.std(corr_flat[valid_post]))

    return (np.array(bias_list), np.array(rmse_list), np.array(crmse_list),
            np.array(bias_list_corr), np.array(rmse_list_corr), np.array(crmse_list_corr),
            mean_obs_list, mean_model_list, mean_corr_list,
            np.array(std_obs_list), np.array(std_model_list), np.array(std_corr_list))


def monthly_correlation(ds_obs, ds_model, mask):
    """Compute monthly correlation within region and average for each year"""
    obs_masked = ds_obs.where(mask)
    model_masked = ds_model.where(mask)
    
    corr_list = []
    for year, group in obs_masked.groupby('time.year'):
        # group is a DataArray with daily values in this year
        monthly_corrs = []
        for month, mgroup in group.groupby('time.month'):
            obs_month = mgroup
            model_month = model_masked.sel(time=mgroup.time)
            
            obs_flat = obs_month.values.flatten()
            model_flat = model_month.values.flatten()
            valid = ~np.isnan(obs_flat) & ~np.isnan(model_flat)
            if np.sum(valid) > 1:
                c = np.corrcoef(obs_flat[valid], model_flat[valid])[0,1]
                monthly_corrs.append(c)
        if len(monthly_corrs) > 0:
            corr_list.append(np.mean(monthly_corrs))
        else:
            corr_list.append(np.nan)
    return np.array(corr_list)


all_stats = {}
for name, mask in regions.items():
    # compute annual metrics + spatial means for boxplots
    (bias, rmse, crmse,
     bias_corr, rmse_corr, crmse_corr,
     mean_obs, mean_prior, mean_post,
     std_obs, std_prior, std_post) = annual_stats(
        ds['obs'], ds['prior'], ds['optimized'], mask
    )
    
    # compute annual correlation for both prior vs obs and posterior vs obs
    corr = monthly_correlation(ds['obs'], ds['prior'], mask)
    corr_corr = monthly_correlation(ds['obs'], ds['optimized'], mask)
    
    all_stats[name] = dict(
        # prior vs obs
        bias=bias,
        rmse=rmse,
        crmse=crmse,
        corr=corr,
        mean_obs=mean_obs,
        mean_prior=mean_prior,
        std_obs=std_obs,
        std_prior=std_prior,
        # posterior vs obs
        bias_post=bias_corr,
        rmse_post=rmse_corr,
        crmse_post=crmse_corr,
        corr_post=corr_corr,
        mean_post=mean_post,
        std_post=std_post
    )

years = np.unique(ds['time.year'].values)


def plot_annual_boxplots_2x2(all_stats, years, figure_title, save_path, colors=None):
    """
    2x2 boxplot figure:
    Top row: model prior/posterior boxes
    Bottom row: observed values corresponding to same years/regions
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    if colors is None:
        colors = ['lightblue','orange','green','red','purple','cyan','gold', 'navy', 'pink', 'brown', 'grey', 'lime']

    n_years = len(years)
    n_regions = len(all_stats)
    width = 0.05
    positions_base = np.arange(n_years)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300, sharey=True)

    # Define which variables to plot
    var_list = [
        ('mean_prior', 'Prior Model'),      # top-left
        ('mean_post', 'Posterior Model'),   # top-right
        ('mean_obs', 'Observed'),           # bottom-left
        ('mean_obs', 'Observed')            # bottom-right (optional duplicate for symmetry)
    ]

    # Compute global min/max for all variables
    all_data = []
    for var_name, _ in var_list:
        for stats in all_stats.values():
            all_data.extend([np.ravel(stats[var_name][i]) for i in range(n_years)])
    global_min = np.min([np.min(d) for d in all_data])
    global_max = np.max([np.max(d) for d in all_data])

    # Plot each subplot
    for idx, (var_name, title) in enumerate(var_list):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        for k, (region_name, stats) in enumerate(all_stats.items()):
            data_per_year = [np.ravel(stats[var_name][i]) for i in range(n_years)]
            pos = positions_base + (k - n_regions/2)*width + width/2
            ax.boxplot(data_per_year, positions=pos, widths=width*0.9,
                       patch_artist=True, boxprops=dict(facecolor=colors[k], alpha=0.6),
                       medianprops=dict(color='black'))

        ax.set_title(title)
        ax.grid(True)
        ax.set_ylim(global_min, global_max)
        ax.set_xticks(positions_base)
        ax.set_xticklabels(years, rotation=45)
        if col == 0:
            ax.set_ylabel('XCO₂ [ppm]')

    # Legend
    handles = [mpatches.Patch(facecolor=c, label=name, alpha=0.6) for c, name in zip(colors, all_stats.keys())]
    axes[0,1].legend(handles=handles, title='Region', bbox_to_anchor=(1.05,1))

    fig.suptitle(figure_title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

plot_annual_boxplots_2x2(
    all_stats=all_stats,
    years=years,
    figure_title='Obs vs Prior/Posterior Annual XCO₂',
    save_path='/home/nponomar/GEOS_Chem_inversion_analysis/Examples/annual_obs_prior_posterior_lc.png'
)

# -----------------------------
# Function definition
# -----------------------------
def plot_stats_bar_2cols(all_stats, years, metrics_keys_prior, metrics_keys_post, titles, colors, filename):
    """
    Plot bar charts for annual statistics in 2 columns:
    Left = prior, Right = post
    Legend below both columns, outside the main figure, slightly closer to x-axis.
    """

    n_years = len(years)
    n_regions = len(all_stats)
    width = 0.05
    n_metrics = len(metrics_keys_prior)

    # Create figure
    fig, axes = plt.subplots(n_metrics, 2, figsize=(10, 8), dpi=300, sharex='col')

    # Loop over metrics
    for row, (metric_prior, metric_post, title) in enumerate(zip(metrics_keys_prior, metrics_keys_post, titles)):
        # Compute y-limits across prior/post
        all_values = []
        for stats in all_stats.values():
            all_values.extend(stats[metric_prior])
            all_values.extend(stats[metric_post])
        y_min = np.min(all_values) * 0.98
        y_max = np.max(all_values) * 1.02
        # Compute mean values
        mean_prior = np.mean([np.mean(stats[metric_prior]) for stats in all_stats.values()])
        mean_post  = np.mean([np.mean(stats[metric_post])  for stats in all_stats.values()])
        for k, (name, stats) in enumerate(all_stats.items()):
            pos = np.arange(n_years) + (k - n_regions/2)*width + width/2

            # Prior
            axes[row, 0].bar(pos, stats[metric_prior], width=width*0.9, color=colors[k], alpha=0.6,
                              label=name if row == 0 else "")
            # Post
            axes[row, 1].bar(pos, stats[metric_post], width=width*0.9, color=colors[k], alpha=0.6,
                              label=name if row == 0 else "")

        # Titles and grid
        axes[row, 0].set_title(f"{title} (Prior, mean={mean_prior:.3f})")
        axes[row, 1].set_title(f"{title} (Post, mean={mean_post:.3f})")
        axes[row, 0].grid(True)
        axes[row, 1].grid(True)
        axes[row, 0].set_ylim(y_min, y_max)
        axes[row, 1].set_ylim(y_min, y_max)

    # x-axis labels
    for ax in axes[-1, :]:
        ax.set_xticks(np.arange(n_years))
        ax.set_xticklabels(years, rotation=45, ha='right')
        ax.set_xlabel('Year')

    # Shrink axes to make room for the legend
    fig.subplots_adjust(bottom=0.22, hspace=0.4, wspace=0.3)

    # Place legend slightly closer to x-axis
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Region', loc='lower center',
               ncol=min(len(all_stats), 4), bbox_to_anchor=(0.5, -0.002))

    # Save figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


# -----------------------------
# Function call
# -----------------------------
metrics_keys_prior = ['bias','rmse','crmse','corr']
metrics_keys_post  = ['bias_post','rmse_post','crmse_post','corr_post']
titles = ['Bias [ppm]','RMSE [ppm]','cRMSE [ppm]','Correlation']
colors = ['lightblue','orange','green','red','purple','cyan','gold', 'navy', 'pink', 'brown', 'grey', 'lime']

plot_stats_bar_2cols(
    all_stats=all_stats,
    years=years,
    metrics_keys_prior=metrics_keys_prior,
    metrics_keys_post=metrics_keys_post,
    titles=titles,
    colors=colors,
    filename='/home/nponomar/GEOS_Chem_inversion_analysis/Examples/annual_stats_prior_post_lc_legend_bottom.png'
)


# -----------------------------
# Function: 2-column Taylor diagram for your all_stats
# -----------------------------
def taylor_diagram_2cols(all_stats, years, colors, filename):
    """
    Two-column Taylor diagram:
    Left = prior vs obs
    Right = posterior vs obs
    """
    n_regions = len(all_stats)
    fig = plt.figure(figsize=(14, 6), dpi=300)

    # Create two polar subplots
    ax_prior = fig.add_subplot(1, 2, 1, polar=True)
    ax_post  = fig.add_subplot(1, 2, 2, polar=True)

    for k, name in enumerate(all_stats.keys()):
        # ------------------
        # Prior vs Obs
        # ------------------
        std_obs = np.array(all_stats[name]['std_obs'])
        std_prior = np.array(all_stats[name]['std_prior'])
        corr = np.array(all_stats[name]['corr'])
        std_prior_norm = std_prior / np.mean(std_obs)
        for i in range(len(years)):
            angle = np.pi/2 - np.arccos(np.clip(corr[i], -1, 1))
            r = std_prior_norm[i]
            ax_prior.plot(angle, r, 'o', color=colors[k], label=name if i==0 else "")

        # ------------------
        # Posterior vs Obs
        # ------------------
        std_post = np.array(all_stats[name]['std_post'])
        corr_post = np.array(all_stats[name]['corr_post'])
        std_post_norm = std_post / np.mean(std_obs)
        for i in range(len(years)):
            angle_post = np.pi/2 - np.arccos(np.clip(corr_post[i], -1, 1))
            r_post = std_post_norm[i]
            ax_post.plot(angle_post, r_post, 'o', color=colors[k], label=name if i==0 else "")
        print(corr_post, corr)
    # Configure axes
    for ax, title in zip([ax_prior, ax_post], ['Prior vs Obs', 'Posterior vs Obs']):
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_thetamin(0)
        ax.set_thetamax(90)
        # Reference circle
        theta = np.linspace(0, np.pi/2, 100)
        ax.plot(theta, np.ones_like(theta), 'k--', label='Reference')
        # Correlation ticks
        theta_ticks = np.radians(np.linspace(0, 90, 7))
        corr_labels = [f"{np.cos(t):.2f}" for t in theta_ticks][::-1]
        ax.set_xticks(theta_ticks)
        ax.set_xticklabels(corr_labels)
        ax.set_rlabel_position(135)
        ax.set_ylim(0, 1.5)
        ax.grid(True)
        ax.set_title(title, fontsize=14)
    for i in range(len(years)):
        print(f"Year {years[i]}: corr_prior={corr[i]:.3f}, theta_prior={np.arccos(corr[i]):.3f} rad")
        print(f"Year {years[i]}: corr_post={corr_post[i]:.3f}, theta_post={np.arccos(corr_post[i]):.3f} rad")
    # Legend below both columns
    handles = [mpatches.Patch(color=c, label=name) for c, name in zip(colors, all_stats.keys())]
    fig.legend(handles=handles, title='Region', loc='lower center', ncol=min(n_regions, 4), bbox_to_anchor=(0.5, -0.05))

    plt.subplots_adjust(top=0.9, bottom=0.18, left=0.05, right=0.95, wspace=0.25)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

# -----------------------------
# Example call
# -----------------------------
colors = ['lightblue','orange','green','red','purple','cyan','gold', 'navy', 'pink', 'brown', 'grey', 'lime']

taylor_diagram_2cols(
    all_stats=all_stats,
    years=years,
    colors=colors,
    filename='/home/nponomar/GEOS_Chem_inversion_analysis/Examples/taylor_prior_post_lc.png'
)


def plot_annual_bias_maps(ds, save_dir, cmap='RdBu_r'):
    """
    Plot annual bias maps (model - obs) for prior and posterior with symmetric color scale
    based on 90th percentile of absolute bias across all years. Maps fill most of the figure.
    Adds mean bias to the title (1 decimal place) and moves colorbar closer to maps.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    os.makedirs(save_dir, exist_ok=True)
    years = np.unique(ds['time.year'].values)
    
    # Compute annual mean biases
    bias_prior_all = (ds['prior'] - ds['obs']).groupby('time.year').mean('time')
    bias_post_all  = (ds['optimized'] - ds['obs']).groupby('time.year').mean('time')
    
    # Determine symmetric vmin/vmax using 90th percentile
    abs_max = np.nanpercentile(np.abs(np.concatenate([bias_prior_all.values.flatten(),
                                                      bias_post_all.values.flatten()])), 90)
    vmin, vmax = -abs_max, abs_max
    
    for y in years:
        ds_year = ds.sel(time=ds['time.year'] == y)
        bias_prior = (ds_year['prior'] - ds_year['obs']).mean(dim='time')
        bias_post  = (ds_year['optimized'] - ds_year['obs']).mean(dim='time')
        
        mean_prior = bias_prior.mean().item()
        mean_post  = bias_post.mean().item()
        
        # Create figure
        fig = plt.figure(figsize=(12,5))
        
        # Manual axes placement for bigger maps
        ax_prior = fig.add_axes([0.05, 0.18, 0.425, 0.75], projection=ccrs.PlateCarree())  # left map
        ax_post  = fig.add_axes([0.525, 0.18, 0.425, 0.75], projection=ccrs.PlateCarree())  # right map
        
        for ax, bias, title, mean_val in zip(
            [ax_prior, ax_post],
            [bias_prior, bias_post],
            ['Prior Bias', 'Posterior Bias'],
            [mean_prior, mean_post]
        ):
            im = ax.pcolormesh(ds['lon'], ds['lat'], bias, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.set_title(f'{title} ({y}), mean={mean_val:.2f} ppm', fontsize=14)
        
        # Shared colorbar below both maps (closer to maps)
        cbar_ax = fig.add_axes([0.15, 0.2, 0.7, 0.03])  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Bias (ppm)')
        
        plt.savefig(f'{save_dir}/Bias_maps_{y}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

plot_annual_bias_maps(ds, save_dir='/home/nponomar/GEOS_Chem_inversion_analysis/Examples/')


#debug Rainforest
# Example: Broadleaf_Tropics mask
mask = regions['Broadleaf_Tropics']

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
im = ax.pcolormesh(ds['lon'], ds['lat'], mask.astype(int), cmap='Greys', vmin=0, vmax=1)
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.set_title('Broadleaf Tropics Mask (1=mask, 0=outside)')
cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.03, pad=0.02)
cbar.set_label('Mask')

plt.savefig(f'/home/nponomar/GEOS_Chem_inversion_analysis/Examples/Broadleaf_Tropics_mask.png', dpi=300, bbox_inches='tight')

