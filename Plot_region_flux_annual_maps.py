import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd

# -------------------------------
# File paths and settings
# -------------------------------
inv_file = '/scratch/local/for_gc_test/enkf_oco2_inv_v14/oco_inv/inv_err_step.nc'
region_file = '/scratch/local/for_gc_test/enkf_oco2_inv_v14/rerun_v14/surface_flux/reg_flux_477_ml.2x2.5.nc'
out_dir = '/home/nponomar/GEOS_Chem_inversion_analysis/Examples/'
start_year = 2018

# -------------------------------
# Load inversion fluxes
# -------------------------------
ds_inv = xr.open_dataset(inv_file)
flux0_da = ds_inv['flux0'].rename({'nt':'time','nx':'layer'})
flux_da  = ds_inv['flux'].rename({'nt':'time','nx':'layer'})

# -------------------------------
# Load region masks
# -------------------------------
ds_region = xr.open_dataset(region_file)
map_da = ds_region['map']  # dims: (lon, lat, layer)

# -------------------------------
# Assign datetime
# -------------------------------
nt = flux0_da.time.size
time = pd.date_range(f'{start_year}-01-01', periods=nt, freq='MS')
flux0_da = flux0_da.assign_coords(time=time)
flux_da  = flux_da.assign_coords(time=time)

# -------------------------------
# Helper to find nearest lon/lat index
# -------------------------------
def nearest_index(coord_array, value):
    return np.abs(coord_array - value).argmin()

# -------------------------------
# Compute annual mean flux maps
# -------------------------------
years = np.unique(time.year)
annual_flux0_maps = []
annual_flux_maps  = []

map_np = map_da.values  # (lon, lat, layer)
lon_vals = map_da.longitude.values
lat_vals = map_da.latitude.values

for yr in years:
    flux0_year = flux0_da.sel(time=str(yr)).mean(dim='time').values
    flux_year  = flux_da.sel(time=str(yr)).mean(dim='time').values

    flux0_map = np.sum(map_np * flux0_year[np.newaxis, np.newaxis, :], axis=2)
    flux_map  = np.sum(map_np * flux_year[np.newaxis, np.newaxis, :], axis=2)

    # Debug prints
    print(f"\nYear {yr}")
    print("flux0_year min/max:", flux0_year.min(), flux0_year.max())
    print("flux_year min/max:", flux_year.min(), flux_year.max())
    print("flux0_map min/max:", flux0_map.min(), flux0_map.max())
    print("flux_map min/max:", flux_map.min(), flux_map.max())

    # Greenland ~ lon=300, lat=75
    lon_idx_g = nearest_index(lon_vals, 300)
    lat_idx_g = nearest_index(lat_vals, 75)
    print("Greenland flux0:", flux0_map[lon_idx_g, lat_idx_g])

    # South Pole ~ lon=0, lat=-90
    lon_idx_s = nearest_index(lon_vals, 0)
    lat_idx_s = nearest_index(lat_vals, -90)
    print("South Pole flux0:", flux0_map[lon_idx_s, lat_idx_s])

    annual_flux0_maps.append(xr.DataArray(flux0_map,
                                         coords={'longitude': lon_vals,
                                                 'latitude': lat_vals},
                                         dims=('longitude','latitude')))
    annual_flux_maps.append(xr.DataArray(flux_map,
                                        coords={'longitude': lon_vals,
                                                'latitude': lat_vals},
                                        dims=('longitude','latitude')))

annual_flux0_maps = xr.concat(annual_flux0_maps, dim='year').assign_coords(year=years)
annual_flux_maps  = xr.concat(annual_flux_maps, dim='year').assign_coords(year=years)

# -------------------------------
# Differences
# -------------------------------
diff_flux = annual_flux_maps - annual_flux0_maps

# Percent difference: 0 where prior flux is exactly 0
diff_percent = xr.where(annual_flux0_maps != 0,
                        100 * diff_flux / annual_flux0_maps,
                        0.0)

# Debug: check min/max
print("diff_percent min/max:", diff_percent.min().values, diff_percent.max().values)

# Debug percent differences
print("\nPercent differences (year 0)")
print("Greenland diff %:", diff_percent.isel(year=0).values[lon_idx_g, lat_idx_g])
print("South Pole diff %:", diff_percent.isel(year=0).values[lon_idx_s, lat_idx_s])

# -------------------------------
# Color scales (NaN-safe)
# -------------------------------
vmin_flux = float(min(annual_flux0_maps.min(), annual_flux_maps.min()))
vmax_flux = float(max(annual_flux0_maps.max(), annual_flux_maps.max()))
vmax_diff = np.nanpercentile(np.abs(diff_flux.values), 99)
vmax_perc = np.nanpercentile(np.abs(diff_percent.values), 80)

# -------------------------------
# Plotting function
# -------------------------------
def plot_map(data, filename, title=None, cmap='RdBu_r', vmin=None, vmax=None, units=None, figsize=(6,5)):
    plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    img = data.plot.pcolormesh(ax=ax, x='longitude', y='latitude', cmap=cmap,
                               vmin=vmin, vmax=vmax, add_colorbar=False, rasterized=True)
    
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.set_global()
    
    if title: plt.title(title, fontsize=10)
    
    cbar = plt.colorbar(img, ax=ax, orientation='horizontal', pad=0.05, fraction=0.05)
    if units: cbar.set_label(units, fontsize=9)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------
# Plot annual maps
# -------------------------------
for i, yr in enumerate(years):
    plot_map(annual_flux0_maps.isel(year=i),
             f'{out_dir}/{yr}_Annual_Prior_CO2_Flux.png',
             title=f'{yr} Annual Prior CO₂ Flux', cmap='viridis',
             vmin=vmin_flux, vmax=vmax_flux, units='PgC/yr')
    
    plot_map(annual_flux_maps.isel(year=i),
             f'{out_dir}/{yr}_Annual_Posterior_CO2_Flux.png',
             title=f'{yr} Annual Posterior CO₂ Flux', cmap='viridis',
             vmin=vmin_flux, vmax=vmax_flux, units='PgC/yr')
    
    plot_map(diff_flux.isel(year=i),
             f'{out_dir}/{yr}_Diff_Flux_PgCyr.png',
             title=f'{yr} Posterior - Prior Flux', cmap='RdBu_r',
             vmin=-vmax_diff, vmax=vmax_diff, units='PgC/yr')
    
    plot_map(diff_percent.isel(year=i),
             f'{out_dir}/{yr}_Diff_Flux_percent.png',
             title=f'{yr} Posterior - Prior Flux (%)', cmap='RdBu_r',
             vmin=-vmax_perc, vmax=vmax_perc, units='%')