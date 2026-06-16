import numpy as np
import netCDF4 as nc
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from pathlib import Path
from functools import partial
from multiprocessing import Pool
import glob
from matplotlib.colors import LogNorm
#Paths
INV_PATH = Path("/scratch/local/enkf_oco2_inv_af/oco_inv")
DATA_PATH = Path("/scratch/local/enkf_oco2_inv_af/oco2_hm_11r_sat_np/2019")
OUTPUT_PATH = Path("/exports/geos.ed.ac.uk/palmer_group/nponomar/GEOS_Chem_inversion_analysis_AF/GEOS_Chem_inversion_analysis/Kalman_gain")

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

#Grid definition
grid_file = Path("/scratch/local/run_co2_af/enkf_output_2019/ST004.EN0047-EN0093/GEOSChem.SatDiagn.20190312_0000z.nc4")

ds_grid = xr.open_dataset(grid_file)

lon_grid = ds_grid.lon.values
lat_grid = ds_grid.lat.values

ds_grid.close()

dlat = np.diff(lat_grid).mean()
dlon = np.diff(lon_grid).mean()

lat_edges = np.concatenate(([lat_grid[0] - dlat / 2],(lat_grid[:-1] + lat_grid[1:]) / 2,[lat_grid[-1] + dlat / 2]))
lon_edges = np.concatenate(([lon_grid[0] - dlon / 2], (lon_grid[:-1] + lon_grid[1:]) / 2, [lon_grid[-1] + dlon / 2]))

n_lat = len(lat_grid)
n_lon = len(lon_grid)


# ============================================================
# READ FUNCTIONS
# ============================================================

def read_step_file(step):
    filepath = INV_PATH / f"std_oco_assim_res.{step:02d}.nc"
    print(f"Reading increment file: {filepath}")
    with nc.Dataset(filepath, 'r') as f:
        dx = f.variables['dx'][:, :]
        NE = len(f.dimensions['ne'])
    return dx, NE


def read_obs_file(date_str):
    filepath = INV_PATH / f"obs_oco_assim_step.{date_str}.nc"
    print(f"Reading observation file: {filepath}")
    with nc.Dataset(filepath, 'r') as f:
        obs = f.variables['obs'][:]
        mod = f.variables['mod'][:] + f.variables['mod_adj'][:]
        hm = f.variables['hm'][:, :]
        lon = f.variables['lon'][:]
        lat = f.variables['lat'][:]
        err = f.variables['err'][:]

    return obs, mod, hm, err, lon, lat


# ============================================================
# KALMAN GAIN PER OBSERVATION
# ============================================================
#a worker function for one date
def process_date_kalman(d, dx, NE, n_lat, n_lon, lat_edges, lon_edges):

    try:
        obs, mod, Y, R, lon, lat = read_obs_file(d)

        nobs = len(obs)

        K_vals = np.zeros(nobs)
        HPHT_vals = np.zeros(nobs)
        Y = Y[:, -NE:]
        # print('Shapes Y, dx, R:', Y.shape, dx.shape, R.shape)
        
        for i in range(nobs):

            y = Y[i, :]

            yy = np.dot(y, y) / (NE - 1)
            xy = np.dot(dx, y) / (NE - 1)

            denom = yy + R[i]

            K_vals[i] = np.linalg.norm(xy / denom)
            HPHT_vals[i] = yy

        K_sum = np.zeros((n_lat, n_lon))
        K_cnt = np.zeros((n_lat, n_lon))

        HPHT_sum = np.zeros((n_lat, n_lon))
        HPHT_cnt = np.zeros((n_lat, n_lon))

        R_sum = np.zeros((n_lat, n_lon))
        R_cnt = np.zeros((n_lat, n_lon))

        # -------------------------
        # GRID INDEXING
        # -------------------------
        lat_idx = np.searchsorted(lat_edges, lat) - 1
        lon_idx = np.searchsorted(lon_edges, lon) - 1

        mask = (
            (lat_idx >= 0) & (lat_idx < n_lat) &
            (lon_idx >= 0) & (lon_idx < n_lon)
        )

        lat_idx = lat_idx[mask]
        lon_idx = lon_idx[mask]

        K_vals_f = K_vals[mask]
        HPHT_vals_f = HPHT_vals[mask]
        R_vals_f = R[mask]

        np.add.at(K_sum, (lat_idx, lon_idx), K_vals_f)
        np.add.at(K_cnt, (lat_idx, lon_idx), 1)

        np.add.at(HPHT_sum, (lat_idx, lon_idx), HPHT_vals_f)
        np.add.at(HPHT_cnt, (lat_idx, lon_idx), 1)

        np.add.at(R_sum, (lat_idx, lon_idx), R_vals_f)
        np.add.at(R_cnt, (lat_idx, lon_idx), 1)

        K_grid = np.full((n_lat, n_lon), np.nan)
        HPHT_grid = np.full((n_lat, n_lon), np.nan)
        R_grid = np.full((n_lat, n_lon), np.nan)

        K_mask = K_cnt > 0
        H_mask = HPHT_cnt > 0
        R_mask = R_cnt > 0

        K_grid[K_mask] = K_sum[K_mask] / K_cnt[K_mask]
        HPHT_grid[H_mask] = HPHT_sum[H_mask] / HPHT_cnt[H_mask]
        R_grid[R_mask] = R_sum[R_mask] / R_cnt[R_mask]
    except:
        print(f"Error processing date {d}. Returning NaN grids.")
        K_grid = np.full((n_lat, n_lon), np.nan)
        HPHT_grid = np.full((n_lat, n_lon), np.nan)
        R_grid = np.full((n_lat, n_lon), np.nan)
        K_vals = np.array([np.nan])

    return {
        "K_grid": K_grid,
        "HPHT_grid": HPHT_grid,
        "R_grid": R_grid,
        "K_mean": np.nanmean(K_vals)
    }


def compute_kalman_step(step, dates):

    dx, NE = read_step_file(step)

    K_maps = []
    HPHT_maps = []
    R_maps = []
    K_series = []

    func = partial(
        process_date_kalman,
        dx=dx,
        NE=NE,
        n_lat=n_lat,
        n_lon=n_lon,
        lat_edges=lat_edges,
        lon_edges=lon_edges
    )

    with Pool(processes=4) as pool:
        results = pool.map(func, dates)

    K_maps = np.array([r["K_grid"] for r in results])
    HPHT_maps = np.array([r["HPHT_grid"] for r in results])
    R_maps = np.array([r["R_grid"] for r in results])
    K_series = np.array([r["K_mean"] for r in results])

    return {
        "step": step,
        "K_map": np.nanmean(K_maps, axis=0),
        "HPHT_map": np.nanmean(HPHT_maps, axis=0),
        "R_map": np.nanmean(R_maps, axis=0),
        "K_series": np.array(K_series)
    }

# ============================================================
# STEP LOOP (same structure as cost function)
# ============================================================

def run_all(step_dates):

    results = []

    for step in sorted(step_dates.keys()):
        print(f"Processing step {step}")
        res = compute_kalman_step(step, step_dates[step])
        results.append(res)

    return results


# ============================================================
# PLOT MAP + TIME SERIES
# ============================================================

def plot_results(results, dates):

    K_mean = np.nanmean([r["K_map"] for r in results], axis=0)

    time_series = np.array([np.nanmean(r["K_series"]) for r in results])
    time_series_median = np.array([np.nanmedian(r["K_series"]) for r in results])
    # -------------------------
    # MAP
    # -------------------------
    
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    im = ax.pcolormesh(
        lon_grid,
        lat_grid,
        K_mean,
        transform=ccrs.PlateCarree(),
        cmap="viridis",
        norm=LogNorm(
            vmin=np.nanpercentile(K_mean, 1),
            vmax=np.nanpercentile(K_mean, 99))
    )
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.8)
    cbar.set_label(r"Mean Kalman Gain ($\mathrm{ppm}^{-1}$)")

    ax.coastlines(linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    plt.savefig(OUTPUT_PATH / "kalman_gain_map.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------
    # TIME SERIES
    # -------------------------

    K_p5 = np.array([np.nanpercentile(r["K_series"], 5) for r in results])
    K_p95 = np.array([np.nanpercentile(r["K_series"], 95) for r in results])

    plt.figure(figsize=(10, 5))
    plt.fill_between(range(len(results)), K_p5, K_p95, alpha=0.25, label="5-95 percentile")
    plt.plot(time_series, marker="o", label="Mean = {:.5f}".format(np.nanmean(time_series)))
    plt.plot(time_series_median, marker="s", label="Median = {:.5f}".format(np.nanmedian(time_series_median)))
    plt.ylabel(r"Kalman Gain ($\mathrm{ppm}^{-1}$)")
    plt.xlabel("Date")
    plt.gca().set_xticklabels(dates, rotation=25)
    plt.grid()
    plt.tight_layout()
    plt.legend()
    plt.savefig(OUTPUT_PATH / "kalman_gain_timeseries.png", dpi=300)
    plt.close()


def plot_gif(results, dates):
    import matplotlib.animation as animation
    K_steps = np.array([r["K_map"] for r in results])

    vmin = max(np.nanpercentile(K_steps, 1), 1e-7)
    vmax = np.nanpercentile(K_steps, 99)
    print(f"Kalman Gain vmin: {vmin}, vmax: {vmax}")
    fig = plt.figure(figsize=(6, 5))

    ax = plt.axes(projection=ccrs.PlateCarree())

    im = ax.pcolormesh(
        lon_grid,
        lat_grid,
        K_steps[0],
        transform=ccrs.PlateCarree(),
        cmap="viridis",
        norm=LogNorm(vmin=vmin, vmax=vmax)
    )

    cbar = plt.colorbar(
        im,
        ax=ax,
        orientation="vertical",
        shrink=0.8
    )
    cbar.set_label(r"Kalman Gain ($\mathrm{ppm}^{-1}$)")

    ax.coastlines(linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False


    title = ax.set_title(f"Kalman Gain step 0")


    def update(frame):

        im.set_array(K_steps[frame].ravel())

        title.set_text(f"Kalman Gain step {results[frame]['step'] + 1} " f"{dates[frame]}")
        return [im, title]


    ani = animation.FuncAnimation(fig,update,frames=len(results),interval=700,blit=False)


    ani.save(OUTPUT_PATH / "kalman_gain_steps.gif",writer="pillow",dpi=150)
    ani.save(OUTPUT_PATH / "kalman_gain_steps.mp4", writer="ffmpeg", dpi=150,fps=2)
    plt.tight_layout()
    plt.close()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    steps = list(range(11))  # replace if needed
    print(f"Processing steps: {steps}")
    def get_dates(step):
        pattern = f"hm.ST{step:03d}*.npz"
        files = glob.glob(str(DATA_PATH / pattern))
        dates = set()
        for f in files:
            b = Path(f).stem
            for p in b.split('.'):
                if len(p) == 8 and p.isdigit():
                    dates.add(p)
        return sorted(dates)

    step_dates = {s: get_dates(s) for s in steps}

    results = run_all(step_dates)

    dates = np.array([step_dates[s][0] for s in steps])
    print(f"Dates for plotting: {dates}")
    plot_results(results, dates)
    plot_gif(results, dates)