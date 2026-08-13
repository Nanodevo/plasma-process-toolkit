# %% [markdown]
# # Tool qualification and chamber matching: releasing Chamber B
#
# A fab rarely runs one chamber. When a second tool is installed, comes
# back from a rebuild, or gets a technology upgrade, it must be
# **qualified** and **matched** to the reference chamber before
# production may split lots across the fleet. This notebook walks that
# release on the simulated PECVD tools:
#
# 1. a qualification block on the candidate chamber (the OQ/PQ logic:
#    does the tool run the recipe stably and capably?),
# 2. a **matching test against the reference chamber** using two
#    one-sided tests (TOST): not "is there a difference?" but "is the
#    difference provably smaller than what the process can tolerate?",
# 3. release into a fleet SPC scheme, or - as happens first here - a
#    corrective action and a re-qualification.
#
# (IQ, the installation qualification, is hardware work - hookup,
# facilities, safety - and is out of scope for a simulation.)
#
# All data is simulated and labeled as such. Run cell by cell.

# %%
import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import ttost_ind

from plasmafab import models, spc

rng = np.random.default_rng(21)
RECIPE = dict(rf_power_w=60.0, pressure_mtorr=550.0,
              sih4_sccm=35.0, temperature_c=250.0)
RESP = "dep_rate_nm_min"


def block(state, n, rng):
    return pd.DataFrame(models.pecvd_run(**RECIPE, state=state, rng=rng)
                        for _ in range(n))

# %% [markdown]
# ## Step 1 - Reference chamber baseline and matching tolerance
#
# The matching tolerance is a process decision, not a statistics
# decision: how much chamber-to-chamber offset can the film spec absorb?
# Here we allow **+-2 %** of the reference mean deposition rate.

# %%
ref_state = models.ToolState()
ref = block(ref_state, 60, rng)
ref_mean = ref[RESP].mean()
delta = 0.02 * ref_mean
print(f"reference: mean {ref_mean:.3f} nm/min, "
      f"matching tolerance +-{delta:.3f}")

# %% [markdown]
# ## Step 2 - Qualification block on Chamber B (and a realistic failure)
#
# Chamber B arrives with a slightly mis-calibrated RF match - the kind
# of small installation debt that a plain "does it run?" check misses
# but a matching test is built to catch.

# %%
chb_state = models.ToolState()
chb_state.rf_match = 0.965          # the hidden installation debt
chb = block(chb_state, 60, rng)
print(f"chamber B: mean {chb[RESP].mean():.3f} nm/min "
      f"({(chb[RESP].mean()/ref_mean - 1)*100:+.1f}% vs reference)")

p_tost, *_ = ttost_ind(ref[RESP], chb[RESP], -delta, +delta)
print(f"TOST equivalence p = {p_tost:.4f} -> "
      f"{'MATCHED' if p_tost < 0.05 else 'NOT matched: hold the tool'}")

# %% [markdown]
# The equivalence test refuses to call the chambers matched: the offset
# is real and larger than the tolerance. Note the logic - a plain t-test
# would ask "different?"; qualification must ask "provably close
# enough?", which is why TOST is the right tool. Chamber B stays on
# engineering hold.
#
# ## Step 3 - Corrective action and re-qualification
#
# The offset traces to the RF delivery (same signature discipline as the
# 8D report: rate low, uniformity slightly up, defects unremarkable).
# After the match network is re-calibrated, the qualification block is
# repeated in full - a re-test, never a waiver.

# %%
chb_state.rf_match = 1.0            # corrective re-calibration
chb2 = block(chb_state, 60, rng)
p_tost2, *_ = ttost_ind(ref[RESP], chb2[RESP], -delta, +delta)
print(f"chamber B after fix: mean {chb2[RESP].mean():.3f} "
      f"({(chb2[RESP].mean()/ref_mean - 1)*100:+.1f}% vs reference)")
print(f"TOST equivalence p = {p_tost2:.4f} -> "
      f"{'MATCHED - release' if p_tost2 < 0.05 else 'still not matched'}")

# %% [markdown]
# ## Step 4 - Release into fleet SPC
#
# Matched is not forever: chambers age apart. Both tools run the same
# chart with the same frozen limits from the reference baseline, so any
# future divergence shows up as one chamber walking away from the other,
# and periodic re-matching is part of the control plan.

# %%
ch = spc.imr_chart(ref[RESP], baseline_n=60)
for name, frame in (("reference", ref), ("chamber B", chb2)):
    inside = ((frame[RESP] > ch["lcl"]) & (frame[RESP] < ch["ucl"])).mean()
    print(f"{name}: {inside:.0%} of qualification runs inside the "
          f"shared frozen limits")

# %% [markdown]
# ## Takeaway
#
# Qualify, prove equivalence, fix and re-qualify when the proof fails,
# then keep the fleet on one chart: tool matching as a statistical
# procedure rather than a gut call. Together with the post-PM
# qualification gate from the 8D report, this is the equipment-release
# half of the quality system in this repository.
#
# ---
# *Method note: OQ/PQ-style qualification logic and TOST equivalence
# testing on simulated tools; data labeled as such. Installation
# qualification (IQ) and vendor-side hardware work are out of scope.*
