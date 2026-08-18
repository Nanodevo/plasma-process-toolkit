# %% [markdown]
# # Tolerancing an optical connection: coupling loss, Monte Carlo, and
# # assembly capability
#
# Optical connectors live or die by mechanical tolerances: a fiber core
# a micron off axis, a ferrule half a degree tilted, a small end-face
# gap - each steals signal. This study walks the standard engineering
# loop for a single-mode butt-coupled connection:
#
# 1. physics: coupling efficiency of two Gaussian modes under lateral,
#    angular and longitudinal misalignment (Marcuse's classic results),
# 2. tolerancing: a Monte Carlo stack-up of realistic assembly
#    tolerances against an insertion-loss budget,
# 3. the manufacturing view: assembly yield expressed as **process
#    capability (Cpk)** with the same `plasmafab.spc` code used for the
#    deposition studies in this repository - because a connector line
#    is a process like any other.
#
# Parameters are SMF-28-like (mode field diameter 10.4 um at 1550 nm).
# All tolerance distributions are assumed and labeled as such.

# %%
import matplotlib.pyplot as plt
import numpy as np

from plasmafab import spc

LAMBDA_UM = 1.55                  # wavelength, um
W0_UM = 10.4 / 2                  # mode field radius, um (SMF-28-like)
Z_R = np.pi * W0_UM**2 / LAMBDA_UM   # Rayleigh range of the fiber mode

def il_db(eta):
    return -10 * np.log10(np.clip(eta, 1e-12, 1.0))

# Marcuse: identical Gaussian modes, small misalignments, air gap
# without Fresnel terms (index matching assumed).
def eta_lateral(d_um):
    return np.exp(-(d_um / W0_UM) ** 2)

def eta_angular(theta_deg):
    th = np.deg2rad(theta_deg)
    return np.exp(-(np.pi * W0_UM * th / LAMBDA_UM) ** 2)

def eta_gap(z_um):
    return 1.0 / (1.0 + (z_um / (2 * Z_R)) ** 2)

# %% [markdown]
# ## 1 - Sensitivities: which misalignment hurts most?

# %%
d = np.linspace(0, 3, 200)
th = np.linspace(0, 2, 200)
z = np.linspace(0, 40, 200)

fig, ax = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
ax[0].plot(d, il_db(eta_lateral(d))); ax[0].set_xlabel("lateral offset (um)")
ax[1].plot(th, il_db(eta_angular(th))); ax[1].set_xlabel("tilt (deg)")
ax[2].plot(z, il_db(eta_gap(z))); ax[2].set_xlabel("end-face gap (um)")
ax[0].set_ylabel("insertion loss (dB)")
for a in ax: a.grid(alpha=0.3)
fig.suptitle("Single-mode coupling loss vs. individual misalignments")
plt.tight_layout()

print(f"loss at 1 um lateral offset : {il_db(eta_lateral(1.0)):.2f} dB")
print(f"loss at 0.5 deg tilt        : {il_db(eta_angular(0.5)):.2f} dB")
print(f"loss at 10 um gap           : {il_db(eta_gap(10.0)):.2f} dB")

# %% [markdown]
# Lateral offset dominates: for a 5.2 um mode radius, one micron costs
# more than a quarter of a dB, while the same micron as an end-face gap
# is nearly free. This asymmetry is why connector mechanics spend their
# precision budget on concentricity, not on gap control.
#
# ## 2 - Monte Carlo tolerance stack-up
#
# Assumed assembly tolerances (labeled, adjustable):
#
# | source | distribution |
# | --- | --- |
# | lateral offset per axis | normal, sigma = 0.7 um |
# | tilt | normal, sigma = 0.4 deg |
# | end-face gap | uniform, 0 to 12 um |
#
# Budget: **IL <= 0.5 dB** per connection (a typical grade spec).

# %%
rng = np.random.default_rng(7)
N = 200_000

dx = rng.normal(0, 0.7, N)
dy = rng.normal(0, 0.7, N)
d_tot = np.hypot(dx, dy)
tilt = np.abs(rng.normal(0, 0.4, N))
gap = rng.uniform(0, 12, N)

eta = eta_lateral(d_tot) * eta_angular(tilt) * eta_gap(gap)
il = il_db(eta)

USL = 0.5
yield_frac = float((il <= USL).mean())
print(f"median IL {np.median(il):.3f} dB, 95th percentile "
      f"{np.percentile(il, 95):.3f} dB")
print(f"assembly yield vs {USL} dB budget: {yield_frac:.1%}")

plt.figure(figsize=(7, 3.2))
plt.hist(il, bins=120, color="#4a7", alpha=0.85)
plt.axvline(USL, color="red", ls="--", label=f"budget {USL} dB")
plt.xlabel("insertion loss (dB)"); plt.ylabel("assemblies")
plt.title("Monte Carlo IL distribution (200k virtual assemblies)")
plt.legend(); plt.tight_layout()

# Variance contribution per source (crude one-at-a-time attribution).
il_lat = il_db(eta_lateral(d_tot))
il_tlt = il_db(eta_angular(tilt))
il_gap = il_db(eta_gap(gap))
tot = il_lat.var() + il_tlt.var() + il_gap.var()
print("variance share - lateral: {:.0%}, tilt: {:.0%}, gap: {:.0%}".format(
    il_lat.var() / tot, il_tlt.var() / tot, il_gap.var() / tot))

# %% [markdown]
# ## 3 - The manufacturing view: capability, not just yield
#
# A yield number says how this batch would do; capability says how much
# margin the *process* has. Treating IL as the response and the 0.5 dB
# budget as a one-sided spec limit, the same capability code that
# judges film uniformity elsewhere in this repository judges the
# connector assembly here. (IL is right-skewed, so Cpk is read as an
# engineering indicator rather than a normal-theory probability - the
# honest caveat a real report would carry.)

# %%
import pandas as pd

cap = spc.capability(pd.Series(il), lsl=None, usl=USL)
print(f"IL mean {cap['mean']:.3f} dB, sigma {cap['sigma']:.3f} dB, "
      f"Cpk vs {USL} dB: {cap['cpk']:.2f}")

# Tolerance-budget experiment: how much does tightening the dominant
# contributor buy? Re-run with lateral sigma reduced 0.7 -> 0.5 um.
dx2 = rng.normal(0, 0.5, N); dy2 = rng.normal(0, 0.5, N)
il2 = il_db(eta_lateral(np.hypot(dx2, dy2)) * eta_angular(tilt)
            * eta_gap(gap))
cap2 = spc.capability(pd.Series(il2), lsl=None, usl=USL)
print(f"lateral sigma 0.5 um -> yield {float((il2 <= USL).mean()):.1%}, "
      f"Cpk {cap2['cpk']:.2f}")

# %% [markdown]
# ## Takeaway
#
# The loop closes the way connector engineering actually runs: physics
# gives the sensitivities, the Monte Carlo turns drawing tolerances
# into an IL distribution, and capability language turns that
# distribution into a manufacturing decision - here, that concentricity
# is the tolerance worth paying for, and by how much.
#
# ---
# *Method note: Gaussian-mode overlap results per Marcuse; SMF-28-like
# parameters; all tolerance distributions assumed and labeled. Written
# as a worked engineering exercise; no proprietary data.*
