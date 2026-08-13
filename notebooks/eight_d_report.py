# %% [markdown]
# # 8D report: deposition-rate excursion after preventive maintenance
#
# The same incident as `root_cause_bad_pm.py`, written up in the format a
# customer or a quality department would actually demand: the **Eight
# Disciplines (8D)** report used across the automotive and semiconductor
# supply chain (and required by IATF 16949-style complaint handling).
#
# The investigation is identical; what this notebook adds is the
# discipline structure around it - containment before root cause,
# verification before closure, prevention before congratulations. Data is
# simulated by the plasmafab tool models and labeled as such throughout.
#
# Run cell by cell (`# %%` works in VS Code and Jupyter).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plasmafab import models, simulate, spc

df = simulate.production_history("pecvd", n_runs=300,
                                 pm_at=150, pm_quality=0.3, seed=6)
resp = "dep_rate_nm_min"
x = df[resp]

# %% [markdown]
# ## D1 - Team
#
# On a real line this is a named cross-functional team with a champion;
# here the roles are played by the modules of this repository:
#
# | Role | Here |
# | --- | --- |
# | Process engineering (lead) | this analysis |
# | SPC / quality             | `plasmafab.spc` |
# | Equipment engineering     | `plasmafab.models.ToolState` (the tool's hidden state) |
# | Production                | the run history itself |
#
# ## D2 - Problem description
#
# **What**: mean deposition rate of the qualified a-Si:H PECVD recipe
# shifted low, with non-uniformity and defect density elevated.
# **Where**: one PECVD chamber. **When**: immediately after the
# preventive maintenance at run 150. **How big**: quantified below.
# **Is / is not**: rate is down on this chamber after this PM; it is not
# a slow drift (step-like onset), and no recipe parameter was changed.

# %%
ch = spc.imr_chart(x)
we = spc.western_electric(x, ch["center"], ch["sigma"])
first_ooc = int(we["index"].min())

before = df.loc[100:149]
after = df.loc[155:210]
d2 = pd.DataFrame({
    "before_mean": before.mean(), "after_mean": after.mean(),
    "shift_%": (after.mean() / before.mean() - 1) * 100,
}).loc[["dep_rate_nm_min", "nonuniformity_pct", "defect_density_au"]]
print(d2.round(2))
print(f"\nfirst Western Electric violation: run {first_ooc} "
      f"(PM was at run 150)")

ax = x.plot(figsize=(10, 3), title="D2 - the problem, quantified")
ax.axhline(ch["ucl"], color="red", lw=0.8)
ax.axhline(ch["lcl"], color="red", lw=0.8)
ax.axvline(150, color="orange", ls="--", label="PM")
ax.axvline(first_ooc, color="purple", ls=":", label="first OOC")
ax.legend(); plt.tight_layout()

# %% [markdown]
# ## D3 - Interim containment
#
# Before any root-cause work: protect the product.
#
# 1. Tool placed on **engineering hold** - no further production release.
# 2. Every wafer processed from the first out-of-control run onward is
#    **quarantined** pending disposition (below: how many that is).
# 3. Downstream: lots already shipped from the suspect window flagged for
#    review.
#
# Containment is deliberately dumb and fast; it buys the time the careful
# work in D4 needs.

# %%
quarantined = df.loc[first_ooc:].index
print(f"quarantined: {len(quarantined)} runs "
      f"(run {quarantined.min()} .. {quarantined.max()})")

# %% [markdown]
# ## D4 - Root cause
#
# The multivariate signature narrows the physics (full walkthrough in
# `root_cause_bad_pm.py`): rate DOWN while non-uniformity and defects are
# UP, all step-like at the PM. Candidate causes and their verdicts:
#
# | Hypothesis | Prediction | Verdict |
# | --- | --- | --- |
# | MFC calibration step | rate moves, little defect signature | rejected |
# | RF match degradation | rate + uniformity, weak defect signal | rejected |
# | Chamber not re-seasoned after PM | rate down, uniformity AND defects up, step at PM | **confirmed** |
#
# Five whys, compressed: rate low -> chamber walls in wrong condition ->
# re-season skipped -> PM checklist does not gate on seasoning -> no
# post-PM qualification step exists. The root cause is **procedural**,
# not a recipe fault.
#
# In the simulation the ground truth is visible directly:

# %%
print(df["wall_factor"].loc[[148, 152]])

# %% [markdown]
# ## D5 - Corrective action (chosen, and what was rejected)
#
# **Chosen**: re-season the chamber per the seasoning recipe, verify with
# a witness run, then release through an SPC gate.
#
# **Considered and rejected**: compensating the rate loss with the R2R
# controller (raise RF power). The controller would indeed pull the rate
# back to target - and silently bake a degraded chamber state into the
# recipe, with the defect density still elevated. Compensation is for
# *normal* drift inside the qualified window, not for masking an
# assignable cause.
#
# ## D6 - Implement and verify
#
# The corrective PM is executed (here: a `preventive_maintenance` with
# `quality=1.0` on a tool state reconstructed at the moment of
# containment), followed by seasoning runs and a verification block that
# must sit inside the frozen baseline limits.

# %%
rng = np.random.default_rng(6)
recipe = df.attrs["recipe"]

# Reconstruct the degraded tool at the containment point, then fix it.
state = models.ToolState()
hist = simulate.production_history  # noqa: F841  (documented path above)
for i in range(160):
    if i == 150:
        state.preventive_maintenance(quality=0.3)   # the bad PM
    models.pecvd_run(**recipe, state=state, rng=rng)
    state.age(1, rng)

state.preventive_maintenance(quality=1.0)           # corrective re-season

verify = pd.Series(
    [models.pecvd_run(**recipe, state=state, rng=rng)["dep_rate_nm_min"]
     for _ in range(40)], name="verification runs")
inside = ((verify > ch["lcl"]) & (verify < ch["ucl"])).mean()
print(f"verification: {len(verify)} runs, "
      f"{inside:.0%} inside frozen baseline limits, "
      f"mean {verify.mean():.2f} vs baseline center {ch['center']:.2f}")

ax = verify.plot(figsize=(10, 2.5), title="D6 - verification block after fix")
ax.axhline(ch["ucl"], color="red", lw=0.8)
ax.axhline(ch["lcl"], color="red", lw=0.8)
ax.axhline(ch["center"], color="green", lw=0.8)
plt.tight_layout()

# %% [markdown]
# ## D7 - Prevent recurrence
#
# The fix repairs the chamber; D7 repairs the *system* that let it
# happen:
#
# 1. **Post-PM qualification gate** added to the tool release procedure:
#    n seasoning runs, one witness wafer, and SPC inside frozen limits
#    before production release. (This is the gate that was missing in the
#    five-why chain.)
# 2. **EWMA monitoring** on the deposition rate (`spc.ewma_chart`): in
#    this incident it alarms several runs before the individuals chart,
#    shrinking any future quarantine window.
# 3. PM checklist and the tool FMEA updated: "chamber seasoning skipped"
#    is now a documented failure mode with a detection control.
#
# ## D8 - Close out
#
# The excursion is closed: contained (D3), explained (D4), fixed and
# verified (D6), and made structurally harder to repeat (D7). On a real
# team this is where the work is acknowledged; in a repository, D8 is the
# commit that ships the gate.
#
# ---
# *Method note: this report follows the standard 8D discipline structure
# as practiced in automotive/semiconductor quality systems. All data is
# generated by the simulated tool models in this repository and labeled
# as such.*
