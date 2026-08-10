import numpy as np
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

base_dir = "/scratch/local/run_co2_af/surface_co2_hm_v9_2016_nc_new_domain/"
files = sorted(glob.glob(base_dir + "**/*.npz", recursive=True))

print(f"Total files found: {len(files)}")

# ---------------------------------------------------------
# 1. Parse metadata + compute diagnostics
# ---------------------------------------------------------

records = []

step_set = set()
ens_set = set()

#domain bounds
LON_MIN, LON_MAX = -40, 75
LAT_MIN, LAT_MAX = -35, 35

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

    t = pd.to_datetime(date.group(1), format="%Y%m%d")
    
    try:
        data = np.load(f)
        obs_lon = data["obs_lon"]
        obs_lat = data["obs_lat"]

        in_domain = (
            (obs_lon >= LON_MIN) & (obs_lon <= LON_MAX) &
            (obs_lat >= LAT_MIN) & (obs_lat <= LAT_MAX)
        )
        # ----------------------------------
        # Ensemble spread
        # hm shape = (n_obs, n_ens)
        # ----------------------------------
        hm = data["hm"]

        hm = hm[in_domain]
        
        # mean response per observation across ensemble members
        hm_obs_mean = np.mean(hm, axis=1)

        mean_hm = np.mean(hm_obs_mean)
        std_hm = np.std(hm_obs_mean)
        if std_hm > 100:
            print(f"Warning: High std_hm ({std_hm}) in file {f}")
        # ----------------------------------
        # Observations
        #
        # Use available observed variable.
        # Common candidates:
        # obs_xco2 / obs / y / obs_apr
        #
        # For now using obs_apr since present
        # in your file keys.
        # Replace later if needed.
        # ----------------------------------
        obs = data["obs"]
        obs = obs[in_domain]
        mean_obs = np.mean(obs)
        std_obs = np.std(obs)

        records.append({
            "file": f,
            "time": t,
            "step": step,
            "ens_block": ens_tag,
            "ens_start": en_start,
            "ens_end": en_end,

            "mean_obs": mean_obs,
            "std_obs": std_obs,

            "mean_hm": mean_hm,
            "std_hm": std_hm,

            "n_obs": hm.shape[0]
        })

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
step_color = {step: cmap(i) for i, step in enumerate(steps)}

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

    for step in steps:
        sub = sub_block[sub_block["step"] == step]

        if len(sub) == 0:
            continue

        color = step_color[step]

        # ----------------------------------
        # Observations:
        # points + error bars
        # ----------------------------------
        ax.errorbar(
            sub["time"],
            sub["mean_obs"],
            yerr=sub["std_obs"],
            fmt="o",
            capsize=2,
            alpha=0.4,
            color=color,
            label=f"ST{step:03d} obs, std / mean hm: {np.mean(sub['std_obs'].values):.2f} / {np.mean(sub['mean_hm'].values):.3f}",
            zorder=1,
        )

        # ----------------------------------
        # Ensemble spread:
        # shaded band around mean_hm
        # ----------------------------------
        ax.fill_between(
            sub["time"],
            sub["mean_obs"] - sub["mean_hm"],
            sub["mean_obs"] + sub["mean_hm"],
            alpha=0.9,
            color=color,
            zorder=2,
        )

    ax.set_title(f"Ensemble block: {block}, median ensemble spread: {np.median(sub_block['mean_hm']):.6f}")
    ax.set_ylabel("XCO2 / hm (ppm)")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8)

# ---------------------------------------------------------
# 5. Shared labels
# ---------------------------------------------------------

axes[-1].set_xlabel("Time")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("/exports/geos.ed.ac.uk/palmer_group/nponomar/GEOS_Chem_inversion_analysis_AF/GEOS_Chem_inversion_analysis/Examples/hm_ensemble2016_nc_new_domain_2016.png", dpi=200)