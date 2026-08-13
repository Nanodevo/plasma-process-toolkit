"""Process FMEA (PFMEA) scaffolding for the virtual tools.

Implements the classic AIAG-style scoring - Severity, Occurrence and
Detection each rated 1..10, multiplied into a Risk Priority Number - plus
the piece a spreadsheet FMEA cannot do: on the simulated tools, detection
ratings can be *measured*. `runs_to_detection` injects a failure mode into
a production history and counts how many runs the monitoring actually
needs to alarm, so the D column is backed by an experiment instead of a
guess.

Everything returns plain pandas so it can feed a notebook or the
dashboard equally well.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import spc


# --------------------------------------------------------------- structure
@dataclass
class FailureMode:
    process_step: str
    failure_mode: str
    effect: str
    severity: int          # 1 (none) .. 10 (safety / line down)
    cause: str
    occurrence: int        # 1 (remote) .. 10 (persistent)
    controls: str          # current detection / prevention controls
    detection: int         # 1 (certain to catch) .. 10 (cannot detect)
    actions: str = ""      # recommended actions, filled during the review

    @property
    def rpn(self) -> int:
        return self.severity * self.occurrence * self.detection


@dataclass
class Pfmea:
    """A process FMEA: a list of failure modes with ranking helpers."""
    process: str
    items: list[FailureMode] = field(default_factory=list)

    def add(self, **kwargs) -> FailureMode:
        fm = FailureMode(**kwargs)
        self.items.append(fm)
        return fm

    def to_frame(self) -> pd.DataFrame:
        rows = [{
            "step": f.process_step, "failure mode": f.failure_mode,
            "effect": f.effect, "S": f.severity, "cause": f.cause,
            "O": f.occurrence, "controls": f.controls, "D": f.detection,
            "RPN": f.rpn, "actions": f.actions,
        } for f in self.items]
        df = pd.DataFrame(rows)
        return df.sort_values("RPN", ascending=False).reset_index(drop=True)

    def top(self, n: int = 3) -> pd.DataFrame:
        return self.to_frame().head(n)


# ------------------------------------------------------- measured detection
def runs_to_detection(history: pd.DataFrame, response: str, event_at: int,
                      baseline_n: int = 60, lam: float = 0.2) -> dict:
    """How many runs after `event_at` until monitoring alarms?

    Evaluates the two monitors a fab would actually run - an individuals
    chart with Western Electric rules and an EWMA chart - on a history
    that contains an injected failure. Returns runs-to-detection for
    each, or None if a monitor never alarms.
    """
    x = history[response]
    ch = spc.imr_chart(x, baseline_n=baseline_n)
    we = spc.western_electric(x, ch["center"], ch["sigma"])
    we_after = we[we["index"] >= event_at]
    imr = int(we_after["index"].min() - event_at) if len(we_after) else None

    ew = spc.ewma_chart(x, lam=lam, baseline_n=baseline_n)
    out = (ew["z"] > ew["ucl"]) | (ew["z"] < ew["lcl"])
    hits = out[out].index[out[out].index >= event_at]
    ewma = int(hits.min() - event_at) if len(hits) else None
    return {"imr_runs": imr, "ewma_runs": ewma}


def detection_rating(runs: int | None, at_risk_per_run: int = 1) -> int:
    """Map measured runs-to-detection onto the 1..10 Detection scale.

    A crude but honest mapping: alarming within a couple of runs is a
    strong control (2-3), within ten runs moderate (4-5), slower than
    that weak (7), and never alarming is a 10. `at_risk_per_run` is a
    reminder that every run before the alarm is product at risk.
    """
    if runs is None:
        return 10
    if runs <= 2:
        return 2
    if runs <= 5:
        return 3
    if runs <= 10:
        return 5
    if runs <= 25:
        return 7
    return 9
