"""Design of Experiments on the virtual plasma tools.

Implements the two workhorse designs used in process engineering, without
external DOE libraries so every step is visible:

* two-level full factorial (with centre points) for screening main effects
  and two-factor interactions,
* face-centred central composite design (CCD) for response-surface
  modelling and process-window mapping.

Model fitting uses ordinary least squares (statsmodels). Factors are coded
to [-1, +1], which makes effect sizes directly comparable, exactly as JMP
or Minitab would report them.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# ------------------------------------------------------------------ designs
def coded_to_real(coded: float, lo: float, hi: float) -> float:
    return lo + (coded + 1.0) * (hi - lo) / 2.0


def full_factorial(factors: dict[str, tuple[float, float]],
                   center_points: int = 4) -> pd.DataFrame:
    """2^k full factorial in coded units, plus replicated centre points."""
    names = list(factors)
    rows = [dict(zip(names, combo))
            for combo in itertools.product([-1.0, 1.0], repeat=len(names))]
    rows += [{n: 0.0 for n in names} for _ in range(center_points)]
    df = pd.DataFrame(rows)
    for n in names:
        lo, hi = factors[n]
        df[n + "_real"] = df[n].map(lambda c: coded_to_real(c, lo, hi))
    return df


def face_centered_ccd(factors: dict[str, tuple[float, float]],
                      center_points: int = 4) -> pd.DataFrame:
    """Face-centred CCD: factorial corners + axial (star) points at the
    face (alpha = 1) + centre points. Supports quadratic models."""
    names = list(factors)
    df = full_factorial(factors, center_points=0)[names]
    axial = []
    for i, n in enumerate(names):
        for level in (-1.0, 1.0):
            row = {m: 0.0 for m in names}
            row[n] = level
            axial.append(row)
    rows = pd.concat([df, pd.DataFrame(axial),
                      pd.DataFrame([{n: 0.0 for n in names}
                                    for _ in range(center_points)])],
                     ignore_index=True)
    for n in names:
        lo, hi = factors[n]
        rows[n + "_real"] = rows[n].map(lambda c: coded_to_real(c, lo, hi))
    return rows


# -------------------------------------------------------------------- runs
def run_design(design: pd.DataFrame, tool_fn, factors, response_keys,
               state=None, rng=None, randomize: bool = True) -> pd.DataFrame:
    """Execute a design on a tool model. Run order is randomized, as it
    would be on a real tool, so slow drift cannot alias with a factor."""
    rng = rng or np.random.default_rng()
    df = design.copy()
    order = rng.permutation(len(df)) if randomize else np.arange(len(df))
    df = df.iloc[order].reset_index(drop=True)
    real_cols = {n: n + "_real" for n in factors}
    results = []
    for _, row in df.iterrows():
        kwargs = {n: row[c] for n, c in real_cols.items()}
        results.append(tool_fn(**kwargs, state=state, rng=rng))
    for key in response_keys:
        df[key] = [r[key] for r in results]
    return df


# ------------------------------------------------------------------- models
def _formula(response: str, names: list[str], quadratic: bool) -> str:
    main = " + ".join(names)
    inter = " + ".join(f"{a}:{b}" for a, b in itertools.combinations(names, 2))
    quad = " + ".join(f"I({n}**2)" for n in names) if quadratic else ""
    parts = [p for p in (main, inter, quad) if p]
    return f"{response} ~ " + " + ".join(parts)


def fit_response(df: pd.DataFrame, response: str, names: list[str],
                 quadratic: bool = False):
    """OLS fit in coded units; returns the fitted statsmodels result."""
    return smf.ols(_formula(response, names, quadratic), data=df).fit()


def effects_table(fit) -> pd.DataFrame:
    """Sorted effect table (coded coefficients, p-values)."""
    out = pd.DataFrame({"coef": fit.params, "p_value": fit.pvalues})
    out = out.drop(index="Intercept", errors="ignore")
    return out.reindex(out.coef.abs().sort_values(ascending=False).index)


def response_surface(fit, names: list[str], x: str, y: str,
                     fixed: dict[str, float] | None = None,
                     n: int = 41):
    """Evaluate the fitted model on a coded 2-D grid for contour plots."""
    fixed = fixed or {}
    g1, g2 = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    grid = pd.DataFrame({x: g1.ravel(), y: g2.ravel()})
    for nme in names:
        if nme not in (x, y):
            grid[nme] = fixed.get(nme, 0.0)
    z = fit.predict(grid).to_numpy().reshape(n, n)
    return g1, g2, z


def process_window(fit_map: dict, names: list[str], x: str, y: str,
                   specs: dict[str, tuple[float | None, float | None]],
                   fixed: dict[str, float] | None = None, n: int = 41):
    """Boolean in-spec map across a 2-D slice of factor space.

    fit_map:  {response: fitted model}
    specs:    {response: (lower_limit_or_None, upper_limit_or_None)}
    Returns (grid_x, grid_y, in_spec_fraction_mask).
    """
    ok = None
    g1 = g2 = None
    for resp, (lo, hi) in specs.items():
        g1, g2, z = response_surface(fit_map[resp], names, x, y, fixed, n)
        mask = np.ones_like(z, dtype=bool)
        if lo is not None:
            mask &= z >= lo
        if hi is not None:
            mask &= z <= hi
        ok = mask if ok is None else (ok & mask)
    return g1, g2, ok
