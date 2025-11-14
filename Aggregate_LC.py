import rioxarray
import xarray as xr
import numpy as np
from xhistogram.xarray import histogram

# Load high-res land cover
land_file = "/exports/geos.ed.ac.uk/palmer_group/nponomar/Landuse/CCI4SEN2COR/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7.tif"
land = rioxarray.open_rasterio(land_file, chunks={'x':2000,'y':2000}).squeeze()

# Keep as DataArray
data = land
if data.name is None:
    data = data.rename("landcover")

y = data['y']
x = data['x']

# Flip y if needed
if y[0] > y[-1]:
    data = data[::-1,:]
    y = y[::-1]

# Load coarse grid
ds_grid = xr.open_dataset("/exports/geos.ed.ac.uk/palmer_group/run_test_2x25/OutputDir/GEOSChem.SatDiagn.20210101_0000z.nc4")
lat_bnds = ds_grid.lat_bnds.values
lon_bnds = ds_grid.lon_bnds.values

n_lat, n_lon = len(ds_grid.lat), len(ds_grid.lon)
lat_bins = np.concatenate([lat_bnds[:,0], [lat_bnds[-1,1]]])
lon_bins = np.concatenate([lon_bnds[:,0], [lon_bnds[-1,1]]])

# Classes to count
classes_to_count = np.array([10, 20, 30, 40, 100, 110, 50, 60,
                             70, 80, 90, 120, 130, 190,
                             200, 220, 140, 150, 160, 170, 180, 210])
classes_sorted = np.sort(classes_to_count)
class_bins = np.concatenate([[classes_sorted[0]-0.5],
                             (classes_sorted[:-1] + classes_sorted[1:])/2,
                             [classes_sorted[-1]+0.5]])

# Make coordinate DataArrays for histogram
class_da = data
lat_da, lon_da = xr.broadcast(y, x)  # lazy broadcasting
lat_da = xr.DataArray(lat_da, dims=['y','x'], coords={'y':y, 'x':x})
lon_da = xr.DataArray(lon_da, dims=['y','x'], coords={'y':y, 'x':x})

# Compute histogram lazily
hist = histogram(class_da, lat_da, lon_da,
                 bins=[class_bins, lat_bins, lon_bins],
                 dim=['y','x'])  # xhistogram expects one dim per argument

hist_np = hist.values
# Top 3 classes
top3_idx = np.argsort(hist_np, axis=0)[-3:][::-1]

# Map indices to actual integer classes
top3_classes = classes_sorted[top3_idx]  # shape (3, n_lat, n_lon)


# top3_classes is your array (3, n_lat, n_lon)
# Use lat/lon from your coarse grid

# Wrap into an xarray DataArray
top3_da = xr.DataArray(
    top3_classes,
    dims=['rank', 'lat', 'lon'],
    coords={
        'rank': [1, 2, 3],
        'lat': ds_grid['lat'].values,
        'lon': ds_grid['lon'].values
    },
    name='top3_landcover'
)

# Save to compressed NetCDF to save space
top3_da.to_netcdf("/exports/geos.ed.ac.uk/palmer_group/nponomar/Landuse/CCI4SEN2COR/top3_lc_2x2p5.nc", encoding={
    'top3_landcover': {'zlib': True, 'complevel': 4, 'dtype': 'int16'}
})



##############Plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# Reclass map
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
outdir = '/home/nponomar/GEOS_Chem_inversion_analysis/Examples/'
# Map top3_classes to 1-9
top3_mapped = np.vectorize(reclass_map.get)(top3_classes)

# Labels for the colorbar
cbar_labels = {
    1: "Broadleaf forest",
    2: "Needleleaf/mixed forest",
    3: "Shrubland",
    4: "Grassland",
    5: "Mosaic vegetation",
    6: "Cropland",
    7: "Urban",
    8: "Water/ocean",
    9: "Other"
}
# Discrete colormap
cmap = plt.get_cmap("tab10", 9)
norm = mcolors.BoundaryNorm(boundaries=np.arange(0.5, 10, 1), ncolors=9)

rank_names = ["Most frequent", "Second", "Third"]
fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

for i in range(3):
    ax = axes[i]
    im = ax.imshow(top3_mapped[i], origin='lower', cmap=cmap, norm=norm)
    ax.set_title(rank_names[i])
    ax.set_xlabel("X grid")
    ax.set_ylabel("Y grid")

# Colorbar with labels
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), ticks=np.arange(1, 10))
cbar.ax.set_yticklabels([cbar_labels[i] for i in range(1, 10)])
cbar.set_label("Land cover type")
plt.savefig(outdir + 'LC_reprojected_regions.png')