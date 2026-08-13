# %% [markdown]
# # Process FMEA for the PECVD deposition step - with measured Detection
#
# A worked **PFMEA** (process failure mode and effects analysis) on the
# virtual PECVD tool, scored AIAG-style: Severity x Occurrence x
# Detection = RPN, each on a 1..10 scale.
#
# The twist that a spreadsheet FMEA cannot offer: on simulated tools the
# **Detection column can be measured, not estimated**. For the failure
# modes our simulator can inject (a bad PM, an MFC calibration step), we
# run the failure and count how many runs the monitoring actually needs
# to alarm - then score D from that experiment. The FMEA then does what
# it is supposed to do as a living document: the top-RPN row gets an
# action, and the row is re-scored with the improved control.
#
# All data is simulated and labeled as such. Run cell by cell.

# %%
import pandas as pd

from plasmafab import fmea, simulate

pd.set_option("display.width", 140)
pd.set_option("display.max_colwidth", 38)

# %% [markdown]
# ## Step 1 - Measure the detection capability of the current controls
#
# Two injectable failure modes, each monitored the way production would:
# an individuals chart with Western Electric rules, and an EWMA chart.

# %%
resp = "dep_rate_nm_min"

bad_pm = simulate.production_history("pecvd", n_runs=300, pm_at=150,
                                     pm_quality=0.3, seed=6)
det_pm = fmea.runs_to_detection(bad_pm, resp, event_at=150)

mfc = simulate.production_history("pecvd", n_runs=300, pm_at=None,
                                  mfc_step_at=150, mfc_step_sccm=-1.5,
                                  seed=6)
det_mfc = fmea.runs_to_detection(mfc, resp, event_at=150)

print("bad PM      :", det_pm)
print("MFC step    :", det_mfc)

# %% [markdown]
# The bad PM is loud (rate, uniformity and defects all move) and the
# chart catches it within a handful of runs. The small MFC step is the
# quiet one: the individuals chart can take far longer than the EWMA,
# which is exactly why the EWMA is part of the control set. Every run
# before the alarm is product at risk, and that is what the D rating has
# to reflect.

# %%
d_pm = fmea.detection_rating(det_pm["imr_runs"])
d_mfc_imr = fmea.detection_rating(det_mfc["imr_runs"])
d_mfc_ewma = fmea.detection_rating(det_mfc["ewma_runs"])
print(f"measured D ratings: bad PM {d_pm}, "
      f"MFC step {d_mfc_imr} (I-MR only) -> {d_mfc_ewma} (with EWMA)")

# %% [markdown]
# ## Step 2 - The PFMEA table
#
# Severity and Occurrence are engineering judgment (as on a real team);
# Detection uses the measured ratings where the simulator can inject the
# mode, and judgment where it cannot (wrong recipe, temperature error).

# %%
p = fmea.Pfmea("PECVD a-Si:H deposition")

p.add(process_step="post-PM release",
      failure_mode="chamber not re-seasoned after PM",
      effect="rate low, defects high; scrap risk on released lots",
      severity=8,
      cause="PM checklist does not gate on seasoning",
      occurrence=4,
      controls="I-MR + Western Electric on rate",
      detection=d_pm)

p.add(process_step="gas delivery",
      failure_mode="MFC calibration step (small)",
      effect="film off-target; slow yield erosion",
      severity=6,
      cause="MFC drift / calibration error",
      occurrence=4,
      controls="I-MR + Western Electric on rate",
      detection=d_mfc_imr)

p.add(process_step="recipe management",
      failure_mode="wrong or edited recipe released",
      effect="entire lot off-spec",
      severity=9,
      cause="manual recipe selection, no change control",
      occurrence=2,
      controls="operator double-check only",
      detection=7)

p.add(process_step="substrate heating",
      failure_mode="temperature setpoint error",
      effect="rate/structure shift (Arrhenius)",
      severity=6,
      cause="thermocouple drift",
      occurrence=3,
      controls="rate SPC (indirect)",
      detection=5)

print(p.to_frame().drop(columns="actions"))

# %% [markdown]
# ## Step 3 - Act on the top risks, then re-score
#
# The FMEA is a to-do list sorted by RPN, not a filing exercise:
#
# 1. **MFC step** (highest measured detection gap): add the EWMA chart to
#    the control plan - detection improves from the I-MR rating to the
#    measured EWMA rating, no new hardware needed.
# 2. **Bad PM**: the 8D report's D7 action (post-PM qualification gate:
#    seasoning runs + witness + SPC gate before release) turns late
#    detection into prevention - occurrence drops.
# 3. **Recipe control**: locked production recipes under change control
#    (see the DMAIC control plan) - occurrence and detection both improve.

# %%
p.items[1].controls += " + EWMA (measured)"
p.items[1].detection = d_mfc_ewma
p.items[1].actions = "EWMA added to control plan"

p.items[0].occurrence = 2
p.items[0].controls += " + post-PM qualification gate"
p.items[0].actions = "8D D7 gate implemented"

p.items[2].occurrence = 1
p.items[2].detection = 3
p.items[2].controls = "locked recipes, change control"
p.items[2].actions = "change control per DMAIC control plan"

after = p.to_frame()
print(after[["step", "failure mode", "S", "O", "D", "RPN", "actions"]])

# %% [markdown]
# ## Result
#
# Every RPN reduction in the re-scored table is traceable either to a
# measured detection experiment (EWMA vs I-MR on the injected MFC step)
# or to a named procedural control that this repository's other
# notebooks build (the 8D qualification gate, the DMAIC change-control
# plan). Risk table -> action -> evidence -> re-score: the FMEA loop,
# closed.
#
# ---
# *Method note: PFMEA structure and 1..10 S/O/D scoring per common AIAG
# practice (the AIAG-VDA Action Priority variant would slot in the same
# way). Data simulated and labeled as such; no certification implied.*
