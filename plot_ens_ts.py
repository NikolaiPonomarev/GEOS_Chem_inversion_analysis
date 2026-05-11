import numpy as np
import glob
import re
import pandas as pd
from collections import defaultdict

base_dir = "/scratch/local/enkf_oco2_inv_af/oco2_hm_11r_sat/2019/"
files = sorted(glob.glob(base_dir + "**/*.npz", recursive=True))

print(f"Total files found: {len(files)}")

# ----------------------------
# 1. Parse all file metadata
# ----------------------------
records = []

step_set = set()
ens_set = set()

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
        hm = data["hm"]#* 1e6 * (28.97 / 44.0)  # convert to ppm

        records.append({
            "file": f,
            "time": t,
            "step": step,
            "ens_block": ens_tag,
            "ens_start": en_start,
            "ens_end": en_end,
            "mean_hm": np.mean(hm),
            "std_hm": np.std(hm),
            "n_obs": hm.shape[0]
        })

    except Exception as e:
        print("Failed:", f, e)

df = pd.DataFrame(records)

# ----------------------------
# 2. Auto-discovered structure
# ----------------------------
steps = sorted(step_set)
ens_blocks = sorted(ens_set)

print("\nDetected structure:")
print("Steps:", steps)
print("Ensemble blocks:", ens_blocks)
print("Total categories:", len(steps) * len(ens_blocks))

# ----------------------------
# 3. Build category labels
# ----------------------------
# df["cat"] = df["step"].astype(str) + "_" + df["ens_block"]

df = df.sort_values("time")

# ----------------------------
# 4. Diagnostics plot
# ----------------------------
import matplotlib.pyplot as plt
import matplotlib.cm as cm

ens_blocks = sorted(df["ens_block"].unique())
steps = sorted(df["step"].unique())



cmap = cm.get_cmap("tab10", len(steps))
step_color = {step: cmap(i) for i, step in enumerate(steps)}

n_blocks = len(ens_blocks)

fig, axes = plt.subplots(
    n_blocks,
    1,
    figsize=(8, 3 * n_blocks),
    sharex=True
)

if n_blocks == 1:
    axes = [axes]  # make iterable

# ----------------------------
# Plot per ensemble block
# ----------------------------
for ax, block in zip(axes, ens_blocks):

    sub_block = df[df["ens_block"] == block]

    for step in steps:
        sub = sub_block[sub_block["step"] == step]

        if len(sub) == 0:
            continue

        ax.errorbar(
            sub["time"],
            sub["mean_hm"],
            yerr=sub["std_hm"],
            fmt="o",
            alpha=0.7,
            capsize=2,
            color=step_color[step],
            label=f"ST{step:03d}"
        )

    ax.set_title(f"Ensemble block: {block}")
    ax.set_ylabel("Mean hm")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.2)

# ----------------------------
# Shared labels
# ----------------------------
axes[-1].set_xlabel("Time")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("hm_diagnostics_blocks.png", dpi=200)
