"""Production simulation: qualified recipes running lot after lot.

Generates the day-to-day data stream SPC watches: a fixed recipe executed
over hundreds of runs while the tool state slowly ages, with optional
injected events (an imperfect PM, an MFC calibration step) that the
monitoring layer is supposed to catch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import models

# Qualified recipes (chosen inside the DOE-derived process windows).
PECVD_RECIPE = dict(rf_power_w=60.0, pressure_mtorr=550.0,
                    sih4_sccm=35.0, temperature_c=250.0)
RIE_RECIPE = dict(rf_power_w=100.0, pressure_mtorr=100.0,
                  sf6_sccm=40.0, bias_v=120.0)
SPUTTER_RECIPE = dict(rf_power_w=150.0, pressure_mtorr=6.0,
                      ar_sccm=60.0, o2_pct=1.5)

_TOOL_FN = {"pecvd": models.pecvd_run, "rie": models.rie_run,
            "sputter": models.sputter_run}
_TOOL_RECIPE = {"pecvd": PECVD_RECIPE, "rie": RIE_RECIPE,
                "sputter": SPUTTER_RECIPE}


def production_history(tool: str = "pecvd",
                       n_runs: int = 300,
                       pm_at: int | None = 150,
                       pm_quality: float = 1.0,
                       mfc_step_at: int | None = None,
                       mfc_step_sccm: float = -1.5,
                       seed: int | None = 6) -> pd.DataFrame:
    """Run a qualified recipe `n_runs` times on an aging tool.

    pm_at / pm_quality    schedule a preventive maintenance; quality < 1
                          simulates a bad PM (e.g. chamber not fully
                          seasoned afterwards) - the root-cause exercise.
    mfc_step_at           a sudden MFC calibration offset at that run.
    """
    rng = np.random.default_rng(seed)
    state = models.ToolState()
    fn = _TOOL_FN[tool]
    recipe = _TOOL_RECIPE[tool]

    rows = []
    for i in range(n_runs):
        if pm_at is not None and i == pm_at:
            state.preventive_maintenance(quality=pm_quality)
        if mfc_step_at is not None and i == mfc_step_at:
            state.mfc_offset += mfc_step_sccm
        out = fn(**recipe, state=state, rng=rng)
        out["run"] = i
        out["rf_match"] = state.rf_match
        out["wall_factor"] = state.wall_factor
        out["mfc_offset"] = state.mfc_offset
        rows.append(out)
        state.age(1, rng)
    df = pd.DataFrame(rows).set_index("run")
    df.attrs["tool"] = tool
    df.attrs["recipe"] = recipe
    df.attrs["events"] = {"pm_at": pm_at, "pm_quality": pm_quality,
                          "mfc_step_at": mfc_step_at}
    return df


def subgrouped(df: pd.DataFrame, response: str, size: int = 5) -> pd.DataFrame:
    """Group a run history into fixed-size subgroups (lots) for X-bar/R."""
    out = df[[response]].copy()
    out["lot"] = out.index // size
    return out.reset_index()


# ---------------------------------------------------------------- cluster
def stack_run(vacuum_break: bool = False,
              pecvd_state: models.ToolState | None = None,
              sputter_state: models.ToolState | None = None,
              rng: np.random.Generator | None = None) -> dict:
    """Deposit a full TCO / p-i-n / TCO stack, as a one-unit cluster tool
    (PECVD + sputter chambers sharing one vacuum system) or as two
    standalone tools with an air break between TCO and silicon layers.

    The one-unit system is how the thesis samples were produced: bottom
    TCO, the a-Si:H p-i-n stack and top TCO without leaving vacuum.
    An air break lets the fresh TCO surface adsorb water/carbon and
    oxidize, which shows up as an interface-defect penalty and a small
    contact-resistance increase - the reason cluster tools exist.
    """
    rng = rng or np.random.default_rng()
    ps = pecvd_state or models.ToolState()
    ss = sputter_state or models.ToolState()

    bottom = models.sputter_run(**SPUTTER_RECIPE, state=ss, rng=rng)
    silicon = models.pecvd_run(**PECVD_RECIPE, state=ps, rng=rng)
    top = models.sputter_run(**SPUTTER_RECIPE, state=ss, rng=rng)

    interface = rng.normal(1.0, 0.1)
    if vacuum_break:
        # Adsorbates + native oxide on the TCO before silicon deposition.
        interface += rng.normal(4.0, 0.8)

    contact_rho = (bottom["resistivity_mohm_cm"]
                   + top["resistivity_mohm_cm"]) / 2.0
    if vacuum_break:
        contact_rho *= rng.normal(1.25, 0.05)

    # Simple stack-quality proxy: silicon bulk defects + interface term,
    # weighted; lower is better.
    quality_penalty = (silicon["defect_density_au"]
                       + 0.8 * interface
                       + 0.3 * contact_rho)

    return {
        "bottom_tco": bottom, "silicon": silicon, "top_tco": top,
        "interface_defects_au": max(0.0, interface),
        "contact_resistivity_mohm_cm": contact_rho,
        "stack_penalty_au": quality_penalty,
        "vacuum_break": vacuum_break,
    }
