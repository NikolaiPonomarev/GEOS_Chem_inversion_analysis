import numpy as np
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


base_dir = "/scratch/local/nponomar/african_inversion_full_setup/gc_05x0625_AS_47L_merra2_CO2/surface_co2_hm_v9_2016_test_nc/"
files = sorted(glob.glob(base_dir + "**/*.npz", recursive=True))

print(f"Total files found: {len(files)}")

block_data = {}   # (ens_block, step) -> list of (lon, lat, spread)

step_set = set()
ens_set = set()

# ---------------------------------------------------------
# Loop over files
# ---------------------------------------------------------

for f in files:

    name = f.split("/")[-1]

    st = re.search(r"ST(\d{3})", name)
    en = re.search(r"EN(\d{4})-EN(\d{4})", name)
    date = re.search(r"(\d{8})", name)

    if not (st and en and date):
        continue

    step = int(st.group(1))
    en_start = int(en.group(1))
    en_end = int(en.group(2))
    ens_tag = f"{en_start:04d}-{en_end:04d}"

    step_set.add(step)
    ens_set.add(ens_tag)

    try:
        data = np.load(f)

        hm = data["hm"]                 # (n_obs, n_ens)
        lon = data["obs_lon"]
        lat = data["obs_lat"]

        # ensemble spread per observation
        spread = np.std(hm, axis=1)

        key = (ens_tag, step)

        if key not in block_data:
            block_data[key] = []

        block_data[key].append((lon, lat, spread))

    except Exception as e:
        print("Failed:", f, e)

# ---------------------------------------------------------
# Sort structure
# ---------------------------------------------------------

steps = sorted(step_set)
blocks = sorted(ens_set)

print("\nDetected structure:")
print("Steps:", steps)
print("Blocks:", blocks)


# ---------------------------------------------------------
# Plot per block/step
# ---------------------------------------------------------
window_size = 16
for (block, step), entries in block_data.items():
    entries = entries[:window_size]
    print(f"\nProcessing Block {block} | Step {step:03d} | Entries: {np.shape(entries[0])}") #files are daily, check the number of obs points for the first files

    lon_all = np.concatenate([e[0] for e in entries])
    lat_all = np.concatenate([e[1] for e in entries])
    sp_all  = np.concatenate([e[2] for e in entries])

    # -----------------------------------------------------
    # Group by exact observation location
    # -----------------------------------------------------
    dfp = pd.DataFrame({
        "lon": lon_all,
        "lat": lat_all,
        "spread": sp_all
    })

    grouped = dfp.groupby(["lon", "lat"], as_index=False).mean()

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------

    fig = plt.figure(figsize=(7, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    sc = ax.scatter(
        grouped["lon"],
        grouped["lat"],
        c=grouped["spread"],
        s=5,
        cmap="viridis",
        transform=ccrs.PlateCarree()
    )

    ax.coastlines(linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.set_extent([-40, 75, -35, 35], crs=ccrs.PlateCarree())
    # ax.set_extent([grouped["lon"].min(), grouped["lon"].max(), grouped["lat"].min(), grouped["lat"].max()], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False

    cb = plt.colorbar(sc, ax=ax, shrink=0.6)
    cb.set_label("Ensemble spread (ppm)")

    plt.title(f"NPZ Ensemble Spread (obs points)\nBlock {block} | ST{step:03d}")

    plt.tight_layout()

    outname = f"/scratch/local/nponomar/african_inversion_full_setup/gc_05x0625_AS_47L_merra2_CO2/plots/npz_spread_map_{block}_ST{step:03d}_2016_test.png"
    plt.savefig(outname, dpi=200)
    plt.close()

    print("Saved:", outname)