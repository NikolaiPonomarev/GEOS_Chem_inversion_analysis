import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature


fpath = '/exports/geos.ed.ac.uk/palmer_group/nponomar/GEOS_Chem_inversion_analysis_AF/code_flux_em/co2_emis_rnd_100_5x5.20190101.nc'

with nc.Dataset(fpath) as ds:
    lon  = ds['longitude'][:]          
    lat  = ds['latitude'][:]           
    smap = ds['map'][:]               


smap = np.transpose(smap, (2, 1, 0))


ens_mean   = smap.mean(axis=0)
ens_spread = smap.std(axis=0)

lat_bounds = {
    'North Africa':   (10,  35),
    'Central Africa': (-10, 10),
    'South Africa':   (-35, -10),
}


region_stats = {}
print("=== Mean scaling factor by sub-domain ===")
print(f"{'Region':<20} {'Mean':>8} {'Spread (std)':>14}")
print("-" * 45)
for region, (lat_lo, lat_hi) in lat_bounds.items():
    mask = (lat >= lat_lo) & (lat <= lat_hi)
    region_vals = smap[:, mask, :]
    r_mean   = float(region_vals.mean())
    r_spread = float(region_vals.std(axis=0).mean())
    region_stats[region] = (lat_lo, lat_hi, r_mean, r_spread)
    print(f"{region:<20} {r_mean:>8.4f} {r_spread:>14.4f}")


fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                         subplot_kw={'projection': ccrs.PlateCarree()})

plot_cfg = [
    (ens_mean,   'Ensemble Mean Scaling Factor',         'RdYlBu_r'),
    (ens_spread, 'Ensemble Spread (std) Scaling Factor', 'YlOrRd'),
]

for ax, (data, title, cmap) in zip(axes, plot_cfg):
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()],
                  crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.4, linestyle=':')
    ax.add_feature(cfeature.LAND,      facecolor='lightgray', zorder=0)

    im = ax.pcolormesh(lon, lat, data, cmap=cmap, transform=ccrs.PlateCarree())
    plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    ax.set_title(title, fontsize=12)

    # draw boundary lines + text labels per region
    for region, (lat_lo, lat_hi, r_mean, r_spread) in region_stats.items():

        # boundary lines
        for lb in [lat_lo, lat_hi]:
            ax.plot([lon.min(), lon.max()], [lb, lb],
                    color='black', linewidth=1.0, linestyle='--',
                    transform=ccrs.PlateCarree())

        # text: centred horizontally, placed at mid-latitude of the band
        lat_mid  = (lat_lo + lat_hi) / 2
        lon_mid  = (lon.min() + lon.max()) / 2
        label    = f"{region}\nμ={r_mean:.3f}  σ={r_spread:.3f}"
        ax.text(lon_mid, lat_mid, label,
                transform=ccrs.PlateCarree(),
                fontsize=8, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=2))

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5)
    gl.top_labels   = False
    gl.right_labels = False

plt.suptitle('CO2 Flux Scaling Factors', fontsize=13)
plt.tight_layout()
plt.savefig('ensemble_scaling_map.png', dpi=150, bbox_inches='tight')

print("Saved: ensemble_scaling_map.png")