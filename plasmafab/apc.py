"""Run-to-run (R2R) process control: the feedback layer above SPC.

SPC detects drift; APC (advanced process control) compensates it. The
workhorse in production is the EWMA run-to-run controller: after each run,
update a smoothed estimate of the process offset and trim one recipe knob
so the next run lands back on target.

Implemented here in its standard single-input single-output form
(Ingolfsson & Sachs style):

    y_k       measured response of run k
    b         process gain, d(response)/d(knob), from the DOE model
    a_k       EWMA offset estimate:  a_k = lam * (y_k - b*u_k) + (1-lam) * a_{k-1}
    u_{k+1}   next recipe setting:   u_{k+1} = (target - a_k) / b

The gain comes from the fitted DOE model - this is the practical reason
fabs run DOE first: the response-surface slope IS the controller gain.

Includes deadband and clamp logic as used on real tools (no adjustment
inside the noise band; never step outside the qualified window).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from . import models


@dataclasses.dataclass
class EwmaR2R:
    """Single-knob EWMA run-to-run controller.

    target      response setpoint (e.g. deposition rate in nm/min)
    gain        process gain b = d(response)/d(knob), in response units
                per knob unit (from DOE / response surface)
    knob0       qualified recipe setting of the manipulated knob
    lam         EWMA forgetting factor (0.2-0.4 typical on real tools)
    deadband    |predicted error| below which no correction is made,
                in response units (avoids chasing noise)
    knob_limits qualified process window for the knob; the controller
                never commands outside it (clamped, flagged)
    """
    target: float
    gain: float
    knob0: float
    lam: float = 0.3
    deadband: float = 0.0
    knob_limits: tuple[float, float] | None = None

    def __post_init__(self):
        # Initialise the offset estimate AT the target: a fresh qualified
        # tool is on target by definition. Starting at zero would make the
        # controller see a huge phantom error on run 1 and slam the knob.
        self.offset = self.target  # a_k, EWMA estimate of process offset
        self.knob = self.knob0
        self.clamped = False

    def update(self, measured: float) -> float:
        """Feed the measurement of the last run; returns the knob value
        to use for the next run."""
        # Update offset estimate from what the model cannot explain.
        self.offset = (self.lam * (measured - self.gain
                                   * (self.knob - self.knob0))
                       + (1.0 - self.lam) * self.offset)
        predicted_err = (self.offset + self.gain
                         * (self.knob - self.knob0)) - self.target
        if abs(predicted_err) <= self.deadband:
            return self.knob
        # Solve for the knob that puts the predicted response on target:
        # target = offset + gain * (knob - knob0)
        self.knob = self.knob0 + (self.target - self.offset) / self.gain
        self.clamped = False
        if self.knob_limits is not None:
            lo, hi = self.knob_limits
            clamped = min(max(self.knob, lo), hi)
            self.clamped = clamped != self.knob
            self.knob = clamped
        return self.knob


def controlled_history(n_runs: int = 300,
                       target: float | None = None,
                       lam: float = 0.3,
                       deadband_sigma: float = 0.5,
                       pm_at: int | None = 150,
                       pm_quality: float = 0.3,
                       seed: int = 2) -> pd.DataFrame:
    """PECVD production with an EWMA R2R controller on RF power.

    Runs the same aging-tool scenario as simulate.production_history
    (including an optionally bad PM) but closes the loop: after each run
    the controller trims RF power to hold deposition rate on target.

    Returns a DataFrame with both the controlled response and the knob
    trajectory, so it can be plotted against the uncontrolled case.
    """
    rng = np.random.default_rng(seed)
    state = models.ToolState()
    recipe = dict(rf_power_w=60.0, pressure_mtorr=550.0,
                  sih4_sccm=35.0, temperature_c=250.0)

    # Controller gain from the local slope of the PECVD rate model:
    # measure d(rate)/d(power) around the recipe with the fresh tool.
    probe = models.ToolState()
    r_lo = np.mean([models.pecvd_run(**{**recipe, "rf_power_w": 55.0},
                                     state=probe, rng=rng)["dep_rate_nm_min"]
                    for _ in range(30)])
    r_hi = np.mean([models.pecvd_run(**{**recipe, "rf_power_w": 65.0},
                                     state=probe, rng=rng)["dep_rate_nm_min"]
                    for _ in range(30)])
    gain = (r_hi - r_lo) / 10.0

    # Baseline: qualified-rate target and noise level for the deadband.
    base = [models.pecvd_run(**recipe, state=models.ToolState(),
                             rng=rng)["dep_rate_nm_min"] for _ in range(30)]
    tgt = target if target is not None else float(np.mean(base))
    noise = float(np.std(base, ddof=1))

    ctrl = EwmaR2R(target=tgt, gain=gain, knob0=recipe["rf_power_w"],
                   lam=lam, deadband=deadband_sigma * noise,
                   knob_limits=(40.0, 90.0))

    rows = []
    knob = recipe["rf_power_w"]
    for i in range(n_runs):
        if pm_at is not None and i == pm_at:
            state.preventive_maintenance(quality=pm_quality)
        out = models.pecvd_run(**{**recipe, "rf_power_w": knob},
                               state=state, rng=rng)
        y = out["dep_rate_nm_min"]
        knob = ctrl.update(y)
        rows.append({"run": i, "dep_rate_nm_min": y,
                     "rf_power_w": knob, "offset_est": ctrl.offset,
                     "clamped": ctrl.clamped,
                     "nonuniformity_pct": out["nonuniformity_pct"],
                     "defect_density_au": out["defect_density_au"]})
        state.age(1, rng)
    df = pd.DataFrame(rows).set_index("run")
    df.attrs["target"] = tgt
    df.attrs["gain"] = gain
    df.attrs["noise"] = noise
    return df
