"""Physics-light models of two plasma tools: PECVD deposition and RIE etch.

These are teaching/simulation models, not calibrated digital twins. The
functional forms follow textbook plasma-processing behaviour (Lieberman &
Lichtenberg; Chapman) and hands-on experience with a-Si:H device fabrication:

* PECVD a-Si:H deposition rate rises with RF power and silane flow,
  saturates at high power (gas-depletion regime), and follows a weak
  Arrhenius temperature dependence. Film uniformity degrades when the rate
  is pushed too hard at low pressure.
* RIE etch rate scales with ion flux (RF power) and the energy of ions
  crossing the sheath (bias voltage). Selectivity to photoresist falls as
  bias rises (sputter-dominated regime). Uniformity has a pressure optimum:
  too low starves the edge of radicals, too high makes transport
  non-uniform.

Every run adds measurement noise and slow tool-state drift, so downstream
DOE and SPC modules see realistic, imperfect data. All numbers are in
plausible ranges for a small research/production fab, but they are invented.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

R_GAS = 8.314  # J/(mol K)


@dataclasses.dataclass
class ToolState:
    """Slowly varying state of a chamber; the hidden cause of drift.

    rf_match     RF matching efficiency, 1.0 = freshly tuned
    wall_factor  chamber-wall conditioning, 1.0 = just cleaned/seasoned
    mfc_offset   mass-flow-controller calibration offset in sccm
    """
    rf_match: float = 1.0
    wall_factor: float = 1.0
    mfc_offset: float = 0.0

    def age(self, runs: int = 1, rng: np.random.Generator | None = None):
        """Advance tool wear by `runs` process runs."""
        rng = rng or np.random.default_rng()
        for _ in range(runs):
            self.rf_match = max(0.85, self.rf_match
                                - abs(rng.normal(6e-6, 3e-6)))
            self.wall_factor = max(0.90, self.wall_factor
                                   - abs(rng.normal(1e-5, 0.5e-5)))
            self.mfc_offset += rng.normal(0.0, 1e-4)

    def preventive_maintenance(self, quality: float = 1.0):
        """PM event. quality < 1 models an imperfect PM (a nice fault
        to hunt in the root-cause exercise)."""
        self.rf_match = 0.98 + 0.02 * quality
        self.wall_factor = 0.97 + 0.03 * quality
        self.mfc_offset *= (1.0 - quality)


# --------------------------------------------------------------------- PECVD
def pecvd_run(rf_power_w: float,
              pressure_mtorr: float,
              sih4_sccm: float,
              temperature_c: float,
              state: ToolState | None = None,
              rng: np.random.Generator | None = None) -> dict:
    """One PECVD a-Si:H deposition run.

    Returns film metrics a fab would track: deposition rate (nm/min),
    thickness non-uniformity (%, 1-sigma across the wafer), and a defect
    density proxy (a.u.) standing in for microvoid/particle density.
    """
    state = state or ToolState()
    rng = rng or np.random.default_rng()

    p_eff = rf_power_w * state.rf_match
    flow = max(1e-3, sih4_sccm + state.mfc_offset)

    # Rate: power-driven radical generation, saturating in flow, weak
    # Arrhenius in substrate temperature (Ea ~ 0.1 eV for a-Si:H growth).
    t_k = temperature_c + 273.15
    arrh = math.exp(-0.10 * 1.602e-19 / (1.381e-23 * t_k))
    arrh_ref = math.exp(-0.10 * 1.602e-19 / (1.381e-23 * (250 + 273.15)))
    rate = (0.55 * p_eff ** 0.7
            * (flow / (flow + 25.0))
            * (pressure_mtorr / 500.0) ** 0.25
            * (arrh / arrh_ref)
            * state.wall_factor)

    # Non-uniformity: worse at high rate + low pressure (edge starvation),
    # best near the tool's sweet spot around 500 mTorr.
    nonuni = (1.2
              + 0.012 * max(0.0, rate - 8.0) ** 1.5
              + 0.9 * abs(math.log(pressure_mtorr / 500.0))
              + 2.0 * (1.0 - state.wall_factor))

    # Defect proxy: particles shed as walls load up, plus ion damage at
    # high power per unit pressure.
    defects = (0.5
               + 8.0 * (1.0 - state.wall_factor)
               + 0.004 * p_eff ** 1.2 / (pressure_mtorr / 500.0))

    return {
        "dep_rate_nm_min": max(0.0, rng.normal(rate, 0.02 * rate + 0.02)),
        "nonuniformity_pct": max(0.2, rng.normal(nonuni, 0.05 * nonuni)),
        "defect_density_au": max(0.0, rng.normal(defects, 0.08 * defects)),
    }


# ----------------------------------------------------------------------- RIE
def rie_run(rf_power_w: float,
            pressure_mtorr: float,
            sf6_sccm: float,
            bias_v: float,
            state: ToolState | None = None,
            rng: np.random.Generator | None = None) -> dict:
    """One RIE etch run on a-Si:H with an SF6-based chemistry.

    Returns etch rate (nm/min), etch non-uniformity (%), and selectivity
    to photoresist (dimensionless).
    """
    state = state or ToolState()
    rng = rng or np.random.default_rng()

    p_eff = rf_power_w * state.rf_match
    flow = max(1e-3, sf6_sccm + state.mfc_offset)

    # Chemical component (radical flux) + physical component (ion energy).
    chem = 1.6 * (flow / (flow + 30.0)) * (pressure_mtorr / 100.0) ** 0.3
    phys = 0.035 * p_eff ** 0.6 * math.sqrt(max(bias_v, 1.0))
    rate = (chem * 22.0 + phys * 9.0) * state.wall_factor

    # Selectivity to resist: chemistry-rich etching is selective, ion
    # bombardment is not.
    selectivity = 18.0 * chem / (0.25 + 0.010 * math.sqrt(max(bias_v, 1.0))
                                 * p_eff ** 0.3)

    # Uniformity optimum near 100 mTorr for this (invented) chamber.
    nonuni = (1.5
              + 1.1 * abs(math.log(pressure_mtorr / 100.0))
              + 0.008 * max(0.0, rate - 60.0)
              + 2.0 * (1.0 - state.rf_match))

    return {
        "etch_rate_nm_min": max(0.0, rng.normal(rate, 0.025 * rate + 0.05)),
        "nonuniformity_pct": max(0.2, rng.normal(nonuni, 0.05 * nonuni)),
        "selectivity": max(0.1, rng.normal(selectivity, 0.05 * selectivity)),
    }


PECVD_FACTORS = {
    "rf_power_w": (20.0, 120.0),
    "pressure_mtorr": (200.0, 1200.0),
    "sih4_sccm": (10.0, 60.0),
    "temperature_c": (180.0, 320.0),
}

RIE_FACTORS = {
    "rf_power_w": (30.0, 200.0),
    "pressure_mtorr": (30.0, 300.0),
    "sf6_sccm": (10.0, 80.0),
    "bias_v": (20.0, 400.0),
}


# ------------------------------------------------------------------ sputter
def sputter_run(rf_power_w: float,
                pressure_mtorr: float,
                ar_sccm: float,
                o2_pct: float,
                state: ToolState | None = None,
                rng: np.random.Generator | None = None) -> dict:
    """One RF magnetron sputter run of a TCO layer (AZO-like).

    Returns deposition rate (nm/min), film resistivity (mOhm cm),
    optical transmittance (%) and thickness non-uniformity (%).

    The captured behaviour: rate scales near-linearly with power and
    falls weakly with pressure (gas scattering); resistivity has a
    U-shaped optimum in oxygen content (too little O2 gives absorbing,
    metal-rich films, too much kills carrier concentration);
    transmittance rises with O2 and saturates; uniformity is best at
    moderate pressure where target erosion and scattering balance.
    """
    state = state or ToolState()
    rng = rng or np.random.default_rng()

    p_eff = rf_power_w * state.rf_match
    # The O2 MFC is the smallest-range controller on the tool, so a given
    # absolute calibration error moves the O2 fraction hardest.
    o2 = max(0.0, o2_pct + state.mfc_offset * 0.5)

    rate = (0.28 * p_eff ** 0.9
            * (ar_sccm / (ar_sccm + 40.0))
            / (pressure_mtorr / 5.0) ** 0.2
            * state.wall_factor)

    o2_opt = 1.5
    resistivity = 0.9 * (1.0 + 0.55 * (o2 - o2_opt) ** 2) \
        * (1.0 + 0.002 * max(0.0, 150.0 - p_eff))

    transmittance = min(92.0, 78.0 + 14.0 * (1.0 - math.exp(-o2 / 0.8))
                        - 0.008 * max(0.0, p_eff - 200.0))

    nonuni = (1.4
              + 1.0 * abs(math.log(max(pressure_mtorr, 0.5) / 6.0))
              + 0.004 * max(0.0, rate - 20.0)
              + 2.0 * (1.0 - state.rf_match))

    return {
        "dep_rate_nm_min": max(0.0, rng.normal(rate, 0.02 * rate + 0.02)),
        "resistivity_mohm_cm": max(0.2, rng.normal(resistivity,
                                                   0.025 * resistivity)),
        "transmittance_pct": min(95.0, rng.normal(transmittance, 0.4)),
        "nonuniformity_pct": max(0.2, rng.normal(nonuni, 0.05 * nonuni)),
    }


SPUTTER_FACTORS = {
    "rf_power_w": (50.0, 300.0),
    "pressure_mtorr": (2.0, 20.0),
    "ar_sccm": (20.0, 100.0),
    "o2_pct": (0.0, 4.0),
}
