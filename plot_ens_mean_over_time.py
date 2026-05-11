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
        hm = data["hm"]   # (n_obs, n_ens)

        n_obs, n_ens = hm.shape

        # collapse obs -> per ensemble member
        hm_ens_mean = np.mean(hm, axis=0)   # (n_ens,)
        hm_ens_std  = np.std(hm, axis=0)    # (n_ens,)

        # ----------------------------
        # FIX: GLOBAL ensemble indexing
        # ----------------------------
        for i in range(n_ens):

            global_ens_id = en_start + i   # IMPORTANT FIX

            records.append({
                "file": f,
                "time": t,
                "step": step,
                "ens_block": ens_tag,
                "ens_member": global_ens_id,

                "value": hm_ens_mean[i],
                "std": hm_ens_std[i],

                "n_obs": n_obs
            })

    except Exception as e:
        print("Failed:", f, e)

df = pd.DataFrame(records)

# ----------------------------
# 2. AUTO STRUCTURE
# ----------------------------
steps = sorted(step_set)
ens_blocks = sorted(ens_set)

print("\nDetected structure:")
print("Steps:", steps)
print("Ensemble blocks:", ens_blocks)

# ----------------------------
# 3. FINAL AGGREGATION (THIS IS WHAT YOU ASKED FOR)
# ----------------------------
g = df.groupby("ens_member")

mean_vals = g["value"].mean()
std_vals  = g["value"].std()

df_plot = pd.DataFrame({
    "ens_member": mean_vals.index,
    "mean": mean_vals.values,
    "std": std_vals.values
}).sort_values("ens_member")

# ----------------------------
# 4. FINAL PLOT (correct answer)
# ----------------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))

plt.errorbar(
    df_plot["ens_member"],
    df_plot["mean"],
    yerr=df_plot["std"],
    fmt="o",
    capsize=3
)
overall_mean = df_plot["mean"].mean()
plt.xlabel("Global ensemble member index")
plt.ylabel("Mean hm")
plt.grid(alpha=0.3)
plt.legend([f"Overall mean: {overall_mean:.3f}"])
plt.tight_layout()
plt.savefig("ensemble_member_diagnostics.png", dpi=200)