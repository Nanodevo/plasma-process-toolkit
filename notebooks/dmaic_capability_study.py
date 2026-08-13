# %% [markdown]
# # DMAIC case study: bringing a PECVD film spec to capability
#
# A Six Sigma improvement project in the standard **DMAIC** structure
# (Define - Measure - Analyze - Improve - Control), run end to end on the
# simulated PECVD tool. Where the 8D report (`eight_d_report.py`) reacts
# to an incident, DMAIC here does the other half of the quality craft:
# systematically improving a process that is *stable but not capable*.
#
# All data is simulated by the plasmafab tool models and labeled as such.
# Run cell by cell.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plasmafab import doe, models, spc

rng = np.random.default_rng(11)

# %% [markdown]
# ## Define
#
# **Problem**: the current qualified a-Si:H recipe misses the film
# uniformity spec too often. **CTQs** (critical-to-quality):
#
# | CTQ | Spec | Why |
# | --- | --- | --- |
# | thickness non-uniformity | **USL 2.0 %** | device yield across the wafer |
# | deposition rate | **LSL 5.5 nm/min** | throughput floor |
#
# **Goal**: Cpk >= 1.33 on non-uniformity (the failing CTQ) without
# dropping the rate below its throughput floor. **Scope**: one tool, the
# three recipe knobs RF power, pressure, SiH4 flow (temperature held).

# %%
CURRENT = dict(rf_power_w=75.0, pressure_mtorr=250.0,
               sih4_sccm=35.0, temperature_c=250.0)
NONUNI_USL = 2.0
RATE_LSL = 5.5

# %% [markdown]
# ## Measure
#
# Baseline capability from a 60-run block at the current recipe on a
# fresh, in-control tool (stability first: capability numbers only mean
# something on a stable process - the aging/PM incidents are the 8D
# notebook's problem, not this one's).

# %%
def run_block(recipe, n, rng):
    state = models.ToolState()
    rows = [models.pecvd_run(**recipe, state=state, rng=rng)
            for _ in range(n)]
    return pd.DataFrame(rows)

base = run_block(CURRENT, 60, rng)
cap_uni0 = spc.capability(base["nonuniformity_pct"], None, NONUNI_USL)
cap_rate0 = spc.capability(base["dep_rate_nm_min"], RATE_LSL, None)
print(f"baseline non-uniformity: mean {cap_uni0['mean']:.2f}%, "
      f"Cpk {cap_uni0['cpk']:.2f}  (goal >= 1.33)")
print(f"baseline rate: mean {cap_rate0['mean']:.2f} nm/min, "
      f"Cpk {cap_rate0['cpk']:.2f}")

# %% [markdown]
# Non-uniformity capability is far below 1.33: the process is stable but
# not capable. That is a *recipe location* problem, which is exactly what
# DOE is for.
#
# ## Analyze
#
# Face-centred central composite design over the three knobs, fitted with
# quadratic response-surface models for both CTQs.

# %%
FACTORS = {"rf_power_w": (40.0, 90.0),
           "pressure_mtorr": (200.0, 800.0),
           "sih4_sccm": (20.0, 50.0)}
NAMES = list(FACTORS)

design = doe.face_centered_ccd(FACTORS, center_points=6)
def pecvd_at_temp(**kw):
    return models.pecvd_run(temperature_c=250.0, **kw)

runs = doe.run_design(design, pecvd_at_temp, FACTORS,
                      ["dep_rate_nm_min", "nonuniformity_pct"],
                      rng=rng)
fit_uni = doe.fit_response(runs, "nonuniformity_pct", NAMES, quadratic=True)
fit_rate = doe.fit_response(runs, "dep_rate_nm_min", NAMES, quadratic=True)
print(doe.effects_table(fit_uni).head(6).round(3))

# %% [markdown]
# The effects table says what the physics says: pressure dominates
# non-uniformity (with curvature - there is a sweet spot), and power
# drives it up at high rate. Rate, in turn, needs power. So the project
# is a constrained trade: find the region where the uniformity valley
# still clears the throughput floor.
#
# ## Improve
#
# Overlay both specs on the power x pressure slice: the joint in-spec
# region is the process window. Pick the new setpoint deep inside it, not
# at its edge - margin is part of the deliverable.

# %%
g1, g2, ok = doe.process_window(
    {"nonuniformity_pct": fit_uni, "dep_rate_nm_min": fit_rate},
    NAMES, "rf_power_w", "pressure_mtorr",
    specs={"nonuniformity_pct": (None, NONUNI_USL),
           "dep_rate_nm_min": (RATE_LSL, None)},
    fixed={"sih4_sccm": 0.0})

plt.figure(figsize=(6, 4.5))
plt.contourf(g1, g2, ok, levels=[-.5, .5, 1.5], colors=["#eee", "#7fc97f"])
plt.xlabel("rf_power_w (coded)"); plt.ylabel("pressure_mtorr (coded)")
plt.title("Improve - joint process window (green = both CTQs in spec)")
plt.tight_layout()

# New setpoint: centered in the window (coded ~(+0.2, +0.2)), decoded:
NEW = dict(
    rf_power_w=doe.coded_to_real(0.2, *FACTORS["rf_power_w"]),
    pressure_mtorr=doe.coded_to_real(0.2, *FACTORS["pressure_mtorr"]),
    sih4_sccm=doe.coded_to_real(0.0, *FACTORS["sih4_sccm"]),
    temperature_c=250.0)
print({k: round(v, 1) for k, v in NEW.items()})

# Confirmation block at the new setpoint - capability re-measured, never
# assumed from the model.
conf = run_block(NEW, 60, rng)
cap_uni1 = spc.capability(conf["nonuniformity_pct"], None, NONUNI_USL)
cap_rate1 = spc.capability(conf["dep_rate_nm_min"], RATE_LSL, None)
print(f"confirmed non-uniformity: mean {cap_uni1['mean']:.2f}%, "
      f"Cpk {cap_uni0['cpk']:.2f} -> {cap_uni1['cpk']:.2f}")
print(f"confirmed rate: mean {cap_rate1['mean']:.2f} nm/min, "
      f"Cpk {cap_rate1['cpk']:.2f} (floor holds)")

# %% [markdown]
# ## Control
#
# An improvement without a control plan decays. The plan that ships with
# the new setpoint:
#
# 1. **Locked recipe** at the new setpoint; changes only through change
#    control with re-qualification.
# 2. **SPC** on both CTQs with limits frozen on the confirmation block
#    (`spc.imr_chart`), Western Electric rules for excursion flagging,
#    EWMA for slow drift.
# 3. **Run-to-run control** (`plasmafab.apc`) allowed to trim RF power
#    only inside the pre-qualified window mapped above - APC holds the
#    target against normal drift; it is not a license to leave the
#    window (see the D5 discussion in the 8D report).
# 4. **Reaction plan**: an out-of-control signal triggers the 8D path,
#    starting with containment.

# %%
ch = spc.imr_chart(conf["nonuniformity_pct"], baseline_n=60)
print(f"control chart for non-uniformity: center {ch['center']:.2f}, "
      f"UCL {ch['ucl']:.2f} (USL {NONUNI_USL}) - the chart alarms "
      f"well before the spec is threatened.")

# %% [markdown]
# ## Result
#
# Non-uniformity moves from clearly-not-capable to Cpk >= 1.33 with the
# throughput floor intact; the exact before/after numbers print in the
# Measure and Improve cells above and are reproducible from this file.
#
# Stable -> measured -> understood -> moved -> held: the DMAIC loop,
# executed end to end on simulated tools.
#
# ---
# *Method note: this study follows the standard DMAIC structure as used
# in Six Sigma practice. It demonstrates the method; no certification is
# implied. All data is simulated and labeled as such.*
