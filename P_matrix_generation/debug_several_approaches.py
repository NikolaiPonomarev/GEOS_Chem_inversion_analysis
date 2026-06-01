import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


FILES = {
    "Random\nindependent": "/exports/geos.ed.ac.uk/palmer_group/nponomar/GEOS_Chem_inversion_analysis_AF/code_flux_em/co2_emis_rnd_100.20190101.nc",

    "Random\n8x8 blocks": "/exports/geos.ed.ac.uk/palmer_group/nponomar/GEOS_Chem_inversion_analysis_AF/code_flux_em/co2_emis_rnd_100_5x5.20190101.nc",

    "Spatial corr.\n1000 km": "/exports/geos.ed.ac.uk/palmer_group/nponomar/GEOS_Chem_inversion_analysis_AF/code_flux_em/co2_emis_spatial_corr_100.20190101.nc",

    "SVD-based\nsampling": "/exports/geos.ed.ac.uk/palmer_group/nponomar/GEOS_Chem_inversion_analysis_AF/code_flux_em/co2_emis_svd_100.20190101.nc",
}


LAT_BOUNDS = {
    "North":   (10,  35),
    "Central": (-10, 10),
    "South":   (-35, -10),
}

def load_ensemble_stats(fpath):

    with nc.Dataset(fpath) as ds:

        lon = ds["longitude"][:]
        lat = ds["latitude"][:]

        # expected original shape:
        # (lon, lat, ensemble)
        smap = ds["map"][:]

    # convert to:
    # (ensemble, lat, lon)
    smap = np.transpose(smap, (2, 1, 0))

    ens_mean = smap.mean(axis=0)
    ens_std  = smap.std(axis=0)

    return lon, lat, ens_mean, ens_std, smap


def compute_region_stats(lat, smap):

    stats = {}

    for region, (lat_lo, lat_hi) in LAT_BOUNDS.items():

        mask = (lat >= lat_lo) & (lat <= lat_hi)

        vals = smap[:, mask, :]

        r_mean = float(vals.mean())
        r_std  = float(vals.std(axis=0).mean())

        stats[region] = (lat_lo, lat_hi, r_mean, r_std)

    return stats


def setup_map(ax, lon, lat):

    ax.set_extent(
        [lon.min(), lon.max(), lat.min(), lat.max()],
        crs=ccrs.PlateCarree()
    )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":")
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        color="gray",
        alpha=0.5
    )

    gl.top_labels = False
    gl.right_labels = False


def add_region_annotations(ax, lon, stats):

    lon_mid = 0.5 * (lon.min() + lon.max())

    for region, (lat_lo, lat_hi, r_mean, r_std) in stats.items():

        # dashed latitude boundaries
        for lb in [lat_lo, lat_hi]:

            ax.plot(
                [lon.min(), lon.max()],
                [lb, lb],
                color="black",
                linewidth=0.8,
                linestyle="--",
                transform=ccrs.PlateCarree()
            )

        lat_mid = 0.5 * (lat_lo + lat_hi)

        label = (
            f"{region}\n"
            f"μ={r_mean:.3f}\n"
            f"σ={r_std:.3f}"
        )

        ax.text( lon_mid, lat_mid, label, fontsize=7, ha="center", va="center", transform=ccrs.PlateCarree(), bbox=dict(facecolor="white",alpha=0.65,edgecolor="none", pad=1.5))

results = {}

global_mean_min = +np.inf
global_mean_max = -np.inf

global_std_min = +np.inf
global_std_max = -np.inf

for name, fpath in FILES.items():

    lon, lat, ens_mean, ens_std, smap = load_ensemble_stats(fpath)

    stats = compute_region_stats(lat, smap)

    results[name] = {
        "lon": lon,
        "lat": lat,
        "mean": ens_mean,
        "std": ens_std,
        "stats": stats,
    }

    global_mean_min = min(global_mean_min, ens_mean.min())
    global_mean_max = max(global_mean_max, ens_mean.max())

    global_std_min = min(global_std_min, ens_std.min())
    global_std_max = max(global_std_max, ens_std.max())


ncols = len(FILES)

fig, axes = plt.subplots(
    2,
    ncols,
    figsize=(4.5 * ncols, 9),
    subplot_kw={"projection": ccrs.PlateCarree()}
)

for col, (name, data_dict) in enumerate(results.items()):

    lon   = data_dict["lon"]
    lat   = data_dict["lat"]
    mean  = data_dict["mean"]
    std   = data_dict["std"]
    stats = data_dict["stats"]

    ax = axes[0, col]

    setup_map(ax, lon, lat)

    im_mean = ax.pcolormesh(
        lon,
        lat,
        mean,
        cmap="RdYlBu_r",
        vmin=global_mean_min,
        vmax=global_mean_max,
        transform=ccrs.PlateCarree()
    )

    add_region_annotations(ax, lon, stats)

    ax.set_title(name, fontsize=12, weight="bold")

    if col == 0:
        ax.text(
            -0.12,
            0.5,
            "Ensemble mean",
            rotation=90,
            fontsize=13,
            weight="bold",
            va="center",
            ha="center",
            transform=ax.transAxes
        )

    ax = axes[1, col]

    setup_map(ax, lon, lat)

    im_std = ax.pcolormesh(
        lon,
        lat,
        std,
        cmap="YlOrRd",
        vmin=global_std_min,
        vmax=global_std_max,
        transform=ccrs.PlateCarree()
    )

    add_region_annotations(ax, lon, stats)

    if col == 0:
        ax.text(
            -0.12,
            0.5,
            "Ensemble spread",
            rotation=90,
            fontsize=13,
            weight="bold",
            va="center",
            ha="center",
            transform=ax.transAxes
        )

plt.subplots_adjust(
    left=0.05,
    right=0.88,
    top=0.93,
    bottom=0.06,
    wspace=0.08,
    hspace=0.12
)

# top-row colorbar (ensemble mean)
cax1 = fig.add_axes([0.90, 0.54, 0.015, 0.32])

cbar1 = fig.colorbar(
    im_mean,
    cax=cax1,
    orientation="vertical"
)

cbar1.set_label(
    "Scaling factor mean",
    fontsize=11
)

# bottom-row colorbar (ensemble spread)
cax2 = fig.add_axes([0.90, 0.13, 0.015, 0.32])

cbar2 = fig.colorbar(
    im_std,
    cax=cax2,
    orientation="vertical"
)

cbar2.set_label(
    "Scaling factor standard deviation",
    fontsize=11
)


# plt.suptitle(
#     "Comparison of Ensemble Scaling Factor Generation Methods\n"
#     "for CO$_2$ Flux Inversions over Africa",
#     fontsize=16,
#     weight="bold",
#     y=0.98
# )

# plt.tight_layout(rect=[0, 0, 1, 0.96])

outfile = "ensemble_scaling_factor_comparison.png"

plt.savefig(
    outfile,
    dpi=200,
    bbox_inches="tight"
)

print(f"Saved: {outfile}")