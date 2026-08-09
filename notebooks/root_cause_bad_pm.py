# %% [markdown]
# # Root-cause walkthrough: the excursion after a PM
#
# Scenario, told the way it happens in a fab: a PECVD tool runs a
# qualified a-Si:H recipe for months. At run 150 the chamber gets a
# preventive maintenance. Shortly after, SPC starts flagging the
# deposition rate. Production wants the tool back; our job is to find
# out what actually happened.
#
# The simulation injects a **bad PM** (chamber opened but not fully
# re-seasoned, `pm_quality=0.3`). The analysis below "does not know"
# that - it has to find it, the same way I traced limiting behaviour of
# my thesis devices back to PECVD deposition parameters.
#
# Run cell by cell (VS Code / Jupyter both understand `# %%`).

# %%
import matplotlib.pyplot as plt
import pandas as pd

from plasmafab import simulate, spc

df = simulate.production_history("pecvd", n_runs=300,
                                 pm_at=150, pm_quality=0.3, seed=6)
resp = "dep_rate_nm_min"
x = df[resp]
x.plot(figsize=(10, 3), title="deposition rate, 300 production runs")
plt.axvline(150, color="orange", ls="--", label="PM")
plt.legend(); plt.tight_layout()

# %% [markdown]
# ## Step 1 - Confirm the excursion statistically
#
# Limits frozen on the qualified baseline (first 60 runs). If we
# recomputed limits over everything, the shift would inflate sigma and
# hide itself.

# %%
ch = spc.imr_chart(x)
we = spc.western_electric(x, ch["center"], ch["sigma"])
print(f"center {ch['center']:.2f}, sigma {ch['sigma']:.3f}")
print(f"{len(we)} rule violations, first at run {int(we['index'].min())}")
print(we.groupby('rule').agg(count=('index', 'size'),
                             first_run=('index', 'min')))

# %% [markdown]
# First violation lands within a handful of runs after the PM: the
# event is real, not noise, and its onset is pinned in time. The EWMA
# chart sees it even faster:

# %%
ew = spc.ewma_chart(x, lam=0.2)
out = (ew["z"] > ew["ucl"]) | (ew["z"] < ew["lcl"])
print("EWMA first alarm: run", int(out[out].index[0]))

# %% [markdown]
# ## Step 2 - Characterize the signature
#
# Mean shift or variance change? Which responses moved - rate only, or
# uniformity and defects too? The signature narrows the physics.

# %%
before = df.loc[100:149]
after = df.loc[155:210]
summary = pd.DataFrame({
    "before_mean": before.mean(), "after_mean": after.mean(),
    "shift_%": (after.mean() / before.mean() - 1) * 100,
}).loc[["dep_rate_nm_min", "nonuniformity_pct", "defect_density_au"]]
print(summary.round(2))

# %% [markdown]
# Rate is DOWN a few percent, non-uniformity and defect density are UP,
# all step-like at the PM. That pattern points at chamber condition
# (wall seasoning), not at an MFC drift (which would move rate mostly,
# slowly, and without a defect signature) and not at RF match (which
# would also hit uniformity but not defects this strongly).
#
# ## Step 3 - Test the hypothesis against the ground truth
#
# On a real tool this step is the seasoning run + witness wafer. In the
# simulation we can simply look at the hidden state:

# %%
df[["wall_factor", "rf_match"]].plot(figsize=(10, 3),
                                     title="hidden tool state")
plt.axvline(150, color="orange", ls="--"); plt.tight_layout()
print(df["wall_factor"].loc[[148, 152]])

# %% [markdown]
# `wall_factor` drops sharply at the PM: the chamber came back worse
# than it went in - a re-season was skipped. The fix is procedural, not
# a recipe change: season the chamber, verify with a witness run, and
# add a post-PM qualification gate (n seasoning runs + SPC back inside
# limits) before releasing the tool to production.
#
# ## Takeaway
#
# The chart says *when*, the multivariate signature says *what
# subsystem*, and the qualification gate turns the incident into a
# procedure so it cannot recur silently. Chart -> signature -> physics
# -> procedure: the same loop as my thesis root-cause work, just on a
# tool instead of a device.
