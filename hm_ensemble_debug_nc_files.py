import numpy as np
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

base_dir = "/scratch/local/run_co2_af/enkf_output/"
files = sorted(
    glob.glob(base_dir + "ST*/GEOSChem.SatDiagn.*.nc4")
)

print(f"Total files found: {len(files)}")


# ---------------------------------------------------------
# 1. Parse metadata + compute diagnostics
# ---------------------------------------------------------

records = []

step_set = set()
ens_set = set()

block_spread_maps = {}
lon = None
lat = None

window_size = 16

for f in files:

    # parent folder contains ST/EN info
    parent = f.split("/")[-2]

    st = re.search(r"ST(\d{3})", parent)
    en = re.search(r"EN(\d{4})-EN(\d{4})", parent)
    date = re.search(r"(\d{8})", f)

    if not (st and en and date):
        continue

    step = int(st.group(1))
    en_start = int(en.group(1))
    en_end = int(en.group(2))
    ens_tag = f"{en_start:04d}-{en_end:04d}"

    step_set.add(step)
    ens_set.add(ens_tag)

    t = pd.to_datetime(date.group(1), format="%Y%m%d")

    try:
        print("Processing:", f)
        ds = xr.open_dataset(f)

        if lon is None:
            lon = ds["lon"].values
            lat = ds["lat"].values

        # ----------------------------------
        # Find all ensemble tracer variables
        # ----------------------------------
        tracer_vars = [
            v for v in ds.data_vars
            if "SatDiagnConc_CO2" in v
        ]

        if len(tracer_vars) == 0:
            print("No tracers found:", f)
            continue

        # ----------------------------------
        # Stack tracers into one array
        #
        # shape:
        # (n_tracers, lon, lat, ...)
        # depending on file structure
        # ----------------------------------
        tracer_stack = np.array([
            ds[v].values.squeeze()
            for v in tracer_vars
        ])
        
        # ----------------------------------
        # Ensemble spread:
        # std across tracer dimension
        # ----------------------------------

        surface_stack = tracer_stack[:, 0, :, :] * 1e6   # (ens, lat, lon), convert to ppm

        spread = np.std(surface_stack, axis=0)     # (lat, lon)
        # print("Surface spread shape:", spread.shape)

        key = (ens_tag, step)

        if key not in block_spread_maps:
            block_spread_maps[key] = []

        if len(block_spread_maps[key]) < window_size:
            block_spread_maps[key].append(spread)

        mean_spread = np.nanmean(spread)
        std_spread = np.nanstd(spread)

        mean_conc = np.nanmean(surface_stack)
        std_conc = np.nanstd(surface_stack)

        records.append({
            "file": f,
            "time": t,
            "step": step,
            "ens_block": ens_tag,
            "ens_start": en_start,
            "ens_end": en_end,

            "mean_spread": mean_spread,
            "std_spread": std_spread,

            "mean_conc": mean_conc,
            "std_conc": std_conc,

            "n_tracers": len(tracer_vars)
        })

        ds.close()

    except Exception as e:
        print("Failed:", f, e)


df = pd.DataFrame(records)

# ---------------------------------------------------------
# 2. Auto-discovered structure
# ---------------------------------------------------------

steps = sorted(step_set)
ens_blocks = sorted(ens_set)

print("\nDetected structure:")
print("Steps:", steps)
print("Ensemble blocks:", ens_blocks)
print("Total categories:", len(steps) * len(ens_blocks))

df = df.sort_values("time")


# ---------------------------------------------------------
# 3. Plot setup
# ---------------------------------------------------------

cmap = cm.get_cmap("tab10", len(steps))
step_color = {
    step: cmap(i)
    for i, step in enumerate(steps)
}

n_blocks = len(ens_blocks)

fig, axes = plt.subplots(
    n_blocks,
    1,
    figsize=(10, 3.5 * n_blocks),
    sharex=True
)

if n_blocks == 1:
    axes = [axes]

# ---------------------------------------------------------
# 4. Plot per ensemble block
# ---------------------------------------------------------

for ax, block in zip(axes, ens_blocks):

    sub_block = df[df["ens_block"] == block]
    first_window_median_spread = []
    for step in steps:
        sub = sub_block[sub_block["step"] == step]

        if len(sub) == 0:
            continue

        sub_first_window = sub.head(window_size)
        first_window_median_spread.append(np.nanmedian(sub_first_window["mean_spread"]))
        

        color = step_color[step]

        # ----------------------------------
        # Mean concentration
        # points + std bars
        # ----------------------------------
        ax.errorbar(
            sub["time"],
            sub["mean_conc"],
            yerr=sub["std_spread"],
            fmt="o",
            capsize=2,
            alpha=0.5,
            color=color,
            label=(
                f"ST{step:03d}, "
                f"spread={np.mean(sub['mean_spread']):.6f}"
            ),
            zorder=1,
        )

        # ----------------------------------
        # Ensemble spread as shaded area
        # around mean concentration
        # ----------------------------------
        ax.fill_between(
            sub["time"],
            sub["mean_conc"] - sub["mean_spread"],
            sub["mean_conc"] + sub["mean_spread"],
            alpha=0.25,
            color=color,
            zorder=2,
        )

    ax.set_title(
        f"Ensemble block: {block}, "
        f"1st window median spread: "
        f"{np.mean(first_window_median_spread):.6f}"
    )

    ax.set_ylabel("CO2 (ppm)")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8)

# ---------------------------------------------------------
# 5. Shared labels
# ---------------------------------------------------------

axes[-1].set_xlabel("Time")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "/exports/geos.ed.ac.uk/palmer_group/nponomar/GEOS_Chem_inversion_analysis_AF/GEOS_Chem_inversion_analysis/Examples/model_domain_srfc_ensemble_spread_from_ncfiles_before_sampling_8x8.png",
    dpi=200
)


for (block, step), maps in block_spread_maps.items():

    mean_map = np.nanmean(
        np.array(maps),
        axis=0
    )

    fig = plt.figure(figsize=(6, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    pcm = ax.pcolormesh(
        lon,
        lat,
        mean_map,
        shading="auto",
        transform=ccrs.PlateCarree()
    )

    # country borders
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    # coastlines
    ax.coastlines(linewidth=0.7)

    # optional gridlines
    ax.gridlines(draw_labels=True, alpha=0.3)

    plt.colorbar(
        pcm,
        ax=ax,
        label="Ensemble spread (ppm)",
        shrink=0.5
    )

    plt.title(
        f"{block} | ST{step:03d}\n"
        f"Mean surface spread (first {window_size} days)"
    )

    plt.tight_layout()

    plt.savefig(
        f"/exports/geos.ed.ac.uk/palmer_group/nponomar/"
        f"GEOS_Chem_inversion_analysis_AF/"
        f"GEOS_Chem_inversion_analysis/Examples/"
        f"spread_map_{block}_ST{step:03d}.png",
        dpi=200
    )

    plt.close()