"""Statistical process control for the virtual tools.

Implements the pieces a fab actually uses day to day:

* individuals / moving-range (I-MR) control charts,
* X-bar / R charts for subgrouped data,
* EWMA chart for catching slow drift early,
* Cp / Cpk process-capability indices,
* Western Electric run rules for excursion flagging.

Everything returns plain pandas/numpy so it can feed the Streamlit
dashboard or a notebook equally well.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Control-chart constants for subgroup sizes 2..10 (standard tables).
_D3 = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
_D4 = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924,
       8: 1.864, 9: 1.816, 10: 1.777}
_A2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419,
       8: 0.373, 9: 0.337, 10: 0.308}


# ------------------------------------------------------------------- charts
def imr_chart(x: pd.Series, baseline_n: int = 60) -> dict:
    """Individuals chart with limits frozen on the first `baseline_n`
    points (the qualified-baseline convention, so drift shows up instead
    of inflating the limits)."""
    base = x.iloc[:baseline_n]
    mr = base.diff().abs().dropna()
    sigma = mr.mean() / 1.128 if len(mr) else x.std()
    center = base.mean()
    return {
        "x": x, "center": center, "sigma": sigma,
        "ucl": center + 3 * sigma, "lcl": center - 3 * sigma,
    }


def xbar_r_chart(df: pd.DataFrame, value: str, subgroup: str,
                 baseline_groups: int = 10) -> dict:
    """X-bar / R chart from long-form data (`value` column, `subgroup` id)."""
    g = df.groupby(subgroup)[value]
    xbar, rng = g.mean(), g.max() - g.min()
    n = int(round(g.size().mean()))
    n = min(max(n, 2), 10)
    xb0 = xbar.iloc[:baseline_groups].mean()
    r0 = rng.iloc[:baseline_groups].mean()
    return {
        "xbar": xbar, "r": rng, "n": n,
        "x_center": xb0, "x_ucl": xb0 + _A2[n] * r0,
        "x_lcl": xb0 - _A2[n] * r0,
        "r_center": r0, "r_ucl": _D4[n] * r0, "r_lcl": _D3[n] * r0,
    }


def ewma_chart(x: pd.Series, lam: float = 0.2,
               baseline_n: int = 60) -> dict:
    """EWMA chart; small lambda gives long memory and catches slow drift
    well before an individuals chart alarms."""
    base = x.iloc[:baseline_n]
    mu, sigma = base.mean(), base.std(ddof=1)
    z = np.zeros(len(x))
    prev = mu
    for i, v in enumerate(x):
        prev = lam * v + (1 - lam) * prev
        z[i] = prev
    k = np.arange(1, len(x) + 1)
    width = 3 * sigma * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * k)))
    return {"z": pd.Series(z, index=x.index), "center": mu,
            "ucl": mu + width, "lcl": mu - width}


# --------------------------------------------------------------- capability
def capability(x: pd.Series, lsl: float | None, usl: float | None) -> dict:
    mu, sigma = x.mean(), x.std(ddof=1)
    cp = cpk = None
    if lsl is not None and usl is not None and sigma > 0:
        cp = (usl - lsl) / (6 * sigma)
    parts = []
    if usl is not None and sigma > 0:
        parts.append((usl - mu) / (3 * sigma))
    if lsl is not None and sigma > 0:
        parts.append((mu - lsl) / (3 * sigma))
    if parts:
        cpk = min(parts)
    return {"mean": mu, "sigma": sigma, "cp": cp, "cpk": cpk}


# ------------------------------------------------------------ run rules
def western_electric(x: pd.Series, center: float, sigma: float) -> pd.DataFrame:
    """Classic four Western Electric rules. Returns a DataFrame of
    violations with rule id and the index where each fires."""
    z = (x - center) / sigma if sigma > 0 else x * 0.0
    hits = []

    def add(rule, idx, desc):
        hits.append({"rule": rule, "index": idx, "description": desc})

    for i in range(len(z)):
        w = z.iloc[max(0, i - 8):i + 1]
        # Rule 1: single point beyond 3 sigma
        if abs(z.iloc[i]) > 3:
            add(1, z.index[i], "point beyond 3 sigma")
        # Rule 2: 2 of 3 beyond 2 sigma, same side
        w3 = z.iloc[max(0, i - 2):i + 1]
        if len(w3) == 3 and ((w3 > 2).sum() >= 2 or (w3 < -2).sum() >= 2):
            add(2, z.index[i], "2 of 3 beyond 2 sigma (same side)")
        # Rule 3: 4 of 5 beyond 1 sigma, same side
        w5 = z.iloc[max(0, i - 4):i + 1]
        if len(w5) == 5 and ((w5 > 1).sum() >= 4 or (w5 < -1).sum() >= 4):
            add(3, z.index[i], "4 of 5 beyond 1 sigma (same side)")
        # Rule 4: 9 consecutive points on one side of centre
        if len(w) == 9 and (all(w > 0) or all(w < 0)):
            add(4, z.index[i], "9 consecutive points on one side")

    df = pd.DataFrame(hits)
    if df.empty:
        return df
    # One alarm per excursion, not one per point: a rule that stays in
    # violation on consecutive points is a single event. Keep the first
    # point of each streak (per rule).
    df = df.sort_values(["rule", "index"])
    pos = {idx: k for k, idx in enumerate(x.index)}
    streak_break = df.groupby("rule")["index"].transform(
        lambda s: s.map(pos).diff().fillna(2) > 1)
    return df[streak_break].sort_values("index").reset_index(drop=True)
