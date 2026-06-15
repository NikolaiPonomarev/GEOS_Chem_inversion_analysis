import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pathlib import Path

OUTPUT_PATH = Path("/exports/geos.ed.ac.uk/palmer_group/nponomar/GEOS_Chem_inversion_analysis_AF/GEOS_Chem_inversion_analysis/cost_function")
data = xr.open_dataset(OUTPUT_PATH / "cost_function_data.nc")

steps = data['assim_step'].values
dates = data['date'].values
nobs  = data['nobs'].values
Jo_prior_norm = data['Jo_prior_norm'].values
Jo_post_norm  = data['Jo_post_norm'].values

Jb = data['Jb'].values
NE = data.attrs['NE']
#Compute Jo per step
unique_steps = np.unique(steps)
nobs_step          = np.array([nobs[steps == s].sum()             for s in unique_steps])
Jo_prior_norm_step = 0.5 * np.array([Jo_prior_norm[steps == s].mean()   for s in unique_steps])
Jo_post_norm_step  = 0.5 * np.array([Jo_post_norm[steps == s].mean()    for s in unique_steps])
date_step          = np.array([dates[steps == s][0]               for s in unique_steps])

Jb = 0.5 * Jb/NE #normalize by the number of ensembles

J_total_prior_norm = Jo_prior_norm_step
J_total_post_norm  = Jo_post_norm_step  + Jb



# Mean values
mean_Jo_prior_norm      = np.mean(Jo_prior_norm_step)
mean_Jo_post_norm       = np.mean(Jo_post_norm_step)
mean_Jb_norm            = np.mean(Jb)
mean_J_total_prior_norm = np.mean(J_total_prior_norm)
mean_J_total_post_norm  = np.mean(J_total_post_norm)

#Plotting
fig, ax = plt.subplots(figsize=(12, 7))

ax.scatter(unique_steps, Jo_prior_norm_step,  alpha=0.7, s=60, c='black', zorder=2)
ax.scatter(unique_steps, Jo_post_norm_step, alpha=0.7, s=60, c='black', zorder=2)

ax.plot(unique_steps, Jo_prior_norm_step, linestyle='--', color='black', alpha=0.3,\
 linewidth=1, label=(r'$J_o^{b}=\frac{1}{2 N_{obs}}\sum\left(\frac{y-H(x_b)}{\sigma_o}\right)^2$' f'\nmean={mean_Jo_prior_norm:.3f}'),zorder=1)

ax.plot(unique_steps, Jo_post_norm_step, color='black', alpha=0.3, linewidth=1,\
 label=(r'$J_o^{post}=\frac{1}{2 N_{obs}}\sum\left(\frac{y-H(x_a)}{\sigma_o}\right)^2$' f'\nmean={mean_Jo_post_norm:.3f}'), zorder=1)

ax.plot(unique_steps, Jb, color='red', linewidth=2, \
        label=(r'$J_b=\frac{1}{2 N_{ens}}\sum\left(\frac{x_a-x_b}{\sigma_b}\right)^2' \
        r'=\frac{1}{2 N_{ens}}{inc_m^T inc_m}$' \
        f'\nmean={mean_Jb_norm:.3f}'), zorder=3)

ax.plot(unique_steps, J_total_prior_norm, color='royalblue', linestyle='--', linewidth=1.5,
        label=r'$J_{total}^{b}$' f'\nmean: {mean_J_total_prior_norm:.3f}', zorder=3)
ax.plot(unique_steps, J_total_post_norm, color='royalblue', linestyle='-', linewidth=1.5,
        label=r'$J_{total}^{posterior}$' f'\nmean: {mean_J_total_post_norm:.3f}', zorder=3)

# Observation count annotations
for s, n, y in zip(unique_steps, nobs_step, J_total_post_norm):
    ax.annotate(f'n={n}',
                xy=(s, y), xytext=(0, 5), textcoords='offset points',
                fontsize=8, fontweight='bold', ha='center', alpha=0.9, rotation=45, zorder=4,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='black', linewidth=0.5))

ax.set_xticks(unique_steps)
ax.set_xticklabels(date_step, rotation=45)
ax.set_xlabel('Date (start of assimilation step)', fontsize=12, fontweight='bold')
ax.set_ylabel('Normalized cost function value', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='best', fontsize=10, framealpha=0.9)

plt.tight_layout()
fig.savefig(OUTPUT_PATH / 'cost_function.png', dpi=300, bbox_inches='tight')
plt.close(fig)


#Interpretation of Jo and Jb values
# If normalized Jo is above 0.5:
# the model–observation difference is larger than the observation uncertainty specified inR.
# This can indicate:
# 1) R is underestimated (observation uncertainty too small)
# 2) The inversion cannot fit concentrations due to model issues, e.g.:
#    - boundary condition errors
#    - transport / representation errors
#
# If normalized Jo is less than 0.5:
# the model fits observations better than expected given the assumed uncertainty in R,
# or observations are not strongly constraining the system.
# Possible reasons:
# 1) R is overestimated (observation uncertainty too large)
# 2) The observations do not sufficiently constrain the state vector
#
# Diagnostic for Jo:
# - Compare (model concentrations + ensemble spread) against (observations + the uncertainty envelope (sqrt(R)))
#   to check if prior ensemble already spans observed values within uncertainty
#
# If normalized Jb is above 0.5 :
# analysis increments are large in ensemble space, meaning observations strongly constrain the state and
# the prior uncertainty (P / ensemble spread) is too small.
#
# If Jb is less than 0.5:
# flux/state adjustments are smaller than expected given the assumed prior uncertainty in P
# Possible reasons:
# 1) Prior uncertainty (P / ensemble spread) is too large
# 2) Observations do not constrain enough degrees of freedom in the state vector
#    (underconstrained regions, localization, or state vector design issues, e.g.
#    boundary conditions not included in the state vector)
#
# Diagnostic for Jb:
# - Inspect spatial structure of increments
# - Optionally compute DOFS (degrees of freedom for signal)