#!/usr/bin/env python3
"""End-to-end DOE on the virtual RIE tool.

Screens the four factors with a 2^4 factorial, fits a response-surface
model on a face-centred CCD, and prints the effects tables plus a
process-window summary. Run:  python examples/run_doe.py
"""
import numpy as np

from plasmafab import doe, models

RESPONSES = ["etch_rate_nm_min", "nonuniformity_pct", "selectivity"]
NAMES = list(models.RIE_FACTORS)


def main():
    rng = np.random.default_rng(11)
    state = models.ToolState()

    # --- screening: 2^4 full factorial + centre points -------------------
    design = doe.full_factorial(models.RIE_FACTORS, center_points=4)
    df = doe.run_design(design, models.rie_run, models.RIE_FACTORS,
                        RESPONSES, state=state, rng=rng)
    print(f"Screening design: {len(df)} runs")
    for resp in RESPONSES:
        fit = doe.fit_response(df, resp, NAMES)
        print(f"\n=== {resp}: top effects (coded units) ===")
        print(doe.effects_table(fit).head(6).round(3).to_string())

    # --- response surface: face-centred CCD ------------------------------
    ccd = doe.face_centered_ccd(models.RIE_FACTORS, center_points=4)
    df2 = doe.run_design(ccd, models.rie_run, models.RIE_FACTORS,
                         RESPONSES, state=state, rng=rng)
    fits = {r: doe.fit_response(df2, r, NAMES, quadratic=True)
            for r in RESPONSES}
    print(f"\nCCD: {len(df2)} runs, quadratic models fitted "
          f"(R2: " + ", ".join(f"{r}={fits[r].rsquared:.2f}"
                               for r in RESPONSES) + ")")

    # --- process window ---------------------------------------------------
    specs = {"etch_rate_nm_min": (60.0, None),
             "nonuniformity_pct": (None, 2.6),
             "selectivity": (12.0, None)}
    g1, g2, ok = doe.process_window(fits, NAMES, "rf_power_w", "bias_v",
                                    specs)
    frac = ok.mean() * 100
    print(f"\nProcess window (rate>=60, NU<=2.6%, sel>=12) across "
          f"power x bias slice: {frac:.0f}% of explored space in spec")
    print("Open the dashboard (streamlit run app.py) for the contour maps.")


if __name__ == "__main__":
    main()
