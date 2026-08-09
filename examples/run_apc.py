#!/usr/bin/env python3
"""Run-to-run control demo: the same bad-PM scenario, open loop vs closed.

Generates docs/apc_r2r.png - the uncontrolled tool drifts off target after
an imperfect PM; the EWMA R2R controller trims RF power each run and holds
the deposition rate on target through the same event.

Run:  python examples/run_apc.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plasmafab import apc, simulate, spc

PM_AT, PM_Q, SEED, N = 150, 0.3, 2, 300


def main():
    ctl = apc.controlled_history(n_runs=N, pm_at=PM_AT, pm_quality=PM_Q,
                                 seed=SEED)
    unc = simulate.production_history("pecvd", n_runs=N, pm_at=PM_AT,
                                      pm_quality=PM_Q, seed=SEED)
    tgt = ctl.attrs["target"]

    lsl, usl = tgt * 0.97, tgt * 1.03
    cpk_u = spc.capability(unc["dep_rate_nm_min"].loc[PM_AT:], lsl, usl)["cpk"]
    cpk_c = spc.capability(ctl["dep_rate_nm_min"].loc[PM_AT:], lsl, usl)["cpk"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 5.4), dpi=150, sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]})

    ax1.plot(unc.index, unc["dep_rate_nm_min"], lw=0.8, color="#b0b7bf",
             label="open loop (SPC only)")
    ax1.plot(ctl.index, ctl["dep_rate_nm_min"], lw=0.9, color="#1b3a5b",
             label="EWMA run-to-run control")
    ax1.axhline(tgt, color="green", lw=1, ls=":", label="target")
    ax1.axhspan(lsl, usl, color="green", alpha=0.07)
    ax1.axvline(PM_AT, color="orange", ls="-.", lw=1.2,
                label=f"bad PM (q={PM_Q})")
    ax1.set_ylabel("dep. rate (nm/min)")
    ax1.set_title("Run-to-run control on the virtual PECVD tool: "
                  f"post-PM Cpk {cpk_u:.2f} (open loop) -> {cpk_c:.2f} "
                  "(controlled)")
    ax1.legend(loc="lower left", fontsize=8, ncol=2)

    ax2.plot(ctl.index, ctl["rf_power_w"], lw=1.0, color="#c05621")
    ax2.axvline(PM_AT, color="orange", ls="-.", lw=1.2)
    ax2.set_ylabel("RF power (W)")
    ax2.set_xlabel("run")
    ax2.set_title("controller action: the knob absorbs tool aging and the "
                  "PM step", fontsize=9)

    fig.tight_layout()
    fig.savefig("docs/apc_r2r.png")
    print(f"target {tgt:.2f} nm/min")
    print(f"post-PM mean deviation: open loop "
          f"{unc['dep_rate_nm_min'].loc[PM_AT+10:].mean() - tgt:+.2f}, "
          f"controlled "
          f"{ctl['dep_rate_nm_min'].loc[PM_AT+10:].mean() - tgt:+.2f}")
    print(f"post-PM Cpk (+-3% specs): {cpk_u:.2f} -> {cpk_c:.2f}")
    print("wrote docs/apc_r2r.png")
    print("Note the honest limit: R2R recovers centering; it cannot reduce "
          "the tool's inherent run-to-run noise.")


if __name__ == "__main__":
    main()
