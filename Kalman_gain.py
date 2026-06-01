import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

obs_file = "/scratch/local/enkf_oco2_inv_af/oco_inv/obs_oco_assim_step.20190101.nc"

state_file = "/scratch/local/enkf_oco2_inv_af/oco_inv/std_oco_assim_res.00.nc"


ds_o = xr.open_dataset(obs_file)
ds_s = xr.open_dataset(state_file)

# ============================================================
# OBSERVATIONS
# ============================================================

obs = ds_o["obs"].values

# prior model
mod = ds_o["mod"].values + ds_o["mod_adj"].values

# observation error variance
R = ds_o["err"].values

# ensemble in observation space
# shape = (nobs, ne)
Y = ds_o["hm"].values

lat = ds_o["lat"].values
lon = ds_o["lon"].values

# ============================================================
# STATE ENSEMBLE PERTURBATIONS
# ============================================================

# X matrix in ETKF notation
# shape = (nx, ne)
X = ds_s["dx"].values

nobs, ne = Y.shape
nx, ne2 = X.shape

assert ne == ne2


innovation = obs - mod


# ============================================================
# DIAGONAL-R ETKF (PER OBSERVATION)
# ============================================================

K_norm = np.zeros(nobs)
K_mean = np.zeros(nobs)

for i in range(nobs):

    y = Y[i, :]          # (ne,)

    # ensemble spread in obs space
    yy = np.dot(y, y) / (ne - 1)

    # cross covariance state–obs
    xy = np.dot(X, y) / (ne - 1)   # (nx,)

    # scalar denominator (R assumed diagonal)
    denom = yy + R[i]

    # Kalman gain vector (state space)
    K_i = xy / denom               # (nx,)

    # scalar diagnostics for mapping
    K_norm[i] = np.linalg.norm(K_i)
    K_mean[i] = np.mean(K_i)

# ============================================================
# OBSERVATION SPACE DIAGNOSTICS
# ============================================================

HPHT = np.sum(Y**2, axis=1) / (ne - 1)
R_std = np.sqrt(R)

# ============================================================
# PLOT
# ============================================================

fig = plt.figure(figsize=(8, 6))
proj = ccrs.PlateCarree()

def setup_ax(pos, title):
    ax = plt.subplot(pos, projection=proj)
    ax.set_title(title)
    ax.coastlines(linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    return ax

# ----------------------------
# PANEL 1: INNOVATION
# ----------------------------

ax1 = setup_ax(221, "Innovation (obs - model)")
sc1 = ax1.scatter(lon, lat, c=innovation, s=5,
                  cmap="coolwarm", transform=proj)
plt.colorbar(sc1, ax=ax1, shrink=0.7)

# ----------------------------
# PANEL 2: ENSEMBLE SPREAD
# ----------------------------

ax2 = setup_ax(222, "Ensemble sensitivity (HPHT)")
sc2 = ax2.scatter(lon, lat, c=HPHT, s=5,
                  cmap="magma", transform=proj)
plt.colorbar(sc2, ax=ax2, shrink=0.7)

# ----------------------------
# PANEL 3: OBS ERROR
# ----------------------------

ax3 = setup_ax(223, "Observation uncertainty (σ)")
sc3 = ax3.scatter(lon, lat, c=R_std, s=5,
                  cmap="viridis", transform=proj)
plt.colorbar(sc3, ax=ax3, shrink=0.7)

# ----------------------------
# PANEL 4: KALMAN GAIN norm
# ----------------------------

ax4 = setup_ax(224, "Kalman gain norm")
sc4 = ax4.scatter(lon, lat, c=K_norm, s=5,
                  cmap="plasma", transform=proj)
plt.colorbar(sc4, ax=ax4, shrink=0.7)

# ============================================================
# SAVE
# ============================================================

plt.tight_layout()

plt.savefig(
    "/exports/geos.ed.ac.uk/palmer_group/nponomar/"
    "GEOS_Chem_inversion_analysis_AF/"
    "GEOS_Chem_inversion_analysis/Examples/"
    "kalman_gain_example_etkf.png",
    dpi=200
)

plt.close()

print("DONE")