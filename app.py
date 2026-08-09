#!/usr/bin/env python3
"""plasmafab dashboard.

Three tabs mirroring the daily work of a plasma process & equipment
engineer: live SPC monitoring of a production tool, DOE-based process
window exploration, and process capability.

    streamlit run app.py

All data is simulated by the plasmafab package. Nothing here is real
fab data.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from plasmafab import doe, models, simulate, spc

st.set_page_config(page_title="plasmafab - virtual plasma fab",
                   page_icon=":material/monitoring:", layout="wide")

st.title("plasmafab: virtual plasma fab")
st.caption("PECVD / RIE / sputter tool models + DOE + SPC + run-to-run "
           "control. **All data simulated** - a practice environment for "
           "plasma process & equipment engineering. "
           "Source: github.com/Nanodevo")

TOOLS = {"PECVD (a-Si:H deposition)": "pecvd", "RIE (SF6 etch)": "rie",
         "Sputter (TCO: AZO)": "sputter"}
RESPONSES = {"pecvd": ["dep_rate_nm_min", "nonuniformity_pct",
                       "defect_density_au"],
             "rie": ["etch_rate_nm_min", "nonuniformity_pct",
                     "selectivity"],
             "sputter": ["dep_rate_nm_min", "resistivity_mohm_cm",
                         "transmittance_pct", "nonuniformity_pct"]}
LABELS = {"dep_rate_nm_min": "deposition rate (nm/min)",
          "etch_rate_nm_min": "etch rate (nm/min)",
          "nonuniformity_pct": "non-uniformity (%)",
          "defect_density_au": "defect density (a.u.)",
          "selectivity": "selectivity to resist",
          "resistivity_mohm_cm": "resistivity (mOhm cm)",
          "transmittance_pct": "transmittance (%)"}
FACTORS = {"pecvd": models.PECVD_FACTORS, "rie": models.RIE_FACTORS,
           "sputter": models.SPUTTER_FACTORS}
TOOL_FN = {"pecvd": models.pecvd_run, "rie": models.rie_run,
           "sputter": models.sputter_run}

tab_spc, tab_doe, tab_apc, tab_cap = st.tabs(
    ["Tool monitoring (SPC)", "Process windows (DOE)",
     "Run-to-run control (APC)", "Capability"])


# ------------------------------------------------------------------ helpers
@st.cache_data(show_spinner=False)
def history(tool, n_runs, pm_at, pm_quality, mfc_step_at, mfc_step, seed):
    return simulate.production_history(
        tool, n_runs=n_runs, pm_at=pm_at, pm_quality=pm_quality,
        mfc_step_at=mfc_step_at, mfc_step_sccm=mfc_step, seed=seed)


@st.cache_data(show_spinner="Running DOE on the virtual tool...")
def run_ccd(tool, seed):
    factors = FACTORS[tool]
    fn = TOOL_FN[tool]
    rng = np.random.default_rng(seed)
    design = doe.face_centered_ccd(factors, center_points=4)
    df = doe.run_design(design, fn, factors, RESPONSES[tool], rng=rng)
    return df, factors


def chart_with_limits(x, center, ucl, lcl, title, marks=None, events=None):
    fig = go.Figure()
    fig.add_scatter(x=x.index, y=x.values, mode="lines+markers",
                    name="measurement", marker=dict(size=4),
                    line=dict(width=1))
    for y, name, dash in ((center, "center", "dot"),
                          (ucl, "UCL", "dash"), (lcl, "LCL", "dash")):
        if np.isscalar(y):
            fig.add_hline(y=y, line_dash=dash,
                          annotation_text=name, line_color="gray")
        else:
            fig.add_scatter(x=x.index, y=y, mode="lines", name=name,
                            line=dict(dash=dash, color="gray", width=1))
    if marks is not None and len(marks):
        fig.add_scatter(x=marks, y=x.loc[marks], mode="markers",
                        name="rule violation",
                        marker=dict(color="red", size=8, symbol="x"))
    for run, label in (events or []):
        fig.add_vline(x=run, line_dash="dashdot", line_color="orange",
                      annotation_text=label)
    fig.update_layout(title=title, height=330,
                      margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.2))
    return fig


# ================================================================== SPC tab
with tab_spc:
    with st.expander("What am I looking at? (plain-language guide)",
                     expanded=False):
        st.markdown("""
This tab plays the role of a fab's **tool-monitoring system**. One recipe runs
over and over on the same machine; every dot is the measured result of one
production run. The question the chart answers: *is the tool still behaving
the way it did when we qualified it?*

**The words behind the abbreviations:**

- **SPC — Statistical Process Control.** Watching a repeating process with
  control charts instead of gut feeling. The **center line** is the average of
  the healthy baseline runs; the **UCL/LCL** (upper/lower control limit) sit
  3 standard deviations away. Points are expected to scatter inside them.
- **PM — Preventive Maintenance.** Scheduled service of the machine (open the
  chamber, clean, replace parts). Here you can inject a *bad* PM: quality 1.0
  means perfectly restored, low quality means the chamber came back in worse
  condition (e.g. not re-seasoned) — a classic real-world fault.
- **MFC — Mass Flow Controller.** The valve that doses gas into the chamber.
  A calibration step means it suddenly delivers slightly different flow than
  its display claims.
- **Western Electric rules.** Four classic alarm patterns (e.g. "one point
  beyond 3 sigma", "9 points in a row on one side of center") that catch
  shifts too small to cross the limits outright.
- **Chart types:** **Individuals (I-MR** — individuals & moving range**)**
  plots every single run. **EWMA** (exponentially weighted moving average)
  plots a smoothed running average, which exposes slow drift much earlier.
  **X-bar / R** groups runs into lots of 5 and plots each lot's average
  (X-bar) and range (R) — the classic factory chart.

**How to read it:** with the default bad PM at run 150, the chart should be
quiet before 150 and start flagging (red x marks) a handful of runs after.
Set PM quality to 1.0 and the alarms should disappear — a healthy tool gives
a quiet chart.
""")
    left, right = st.columns([1, 3])
    with left:
        tool_name = st.selectbox("Tool", list(TOOLS), key="spc_tool")
        tool = TOOLS[tool_name]
        resp = st.selectbox("Monitored response",
                            RESPONSES[tool],
                            format_func=lambda r: LABELS[r])
        n_runs = st.slider("Production runs", 100, 600, 300, 50)
        st.markdown("**Inject events** *(faults the monitoring should "
                    "catch)*")
        pm_on = st.checkbox("Preventive maintenance (PM) at run 150",
                            value=True)
        pm_quality = st.slider("PM quality", 0.0, 1.0, 0.3, 0.1,
                               help="1.0 = perfect maintenance. Low quality "
                                    "models a chamber not re-seasoned after "
                                    "opening.")
        mfc_on = st.checkbox("Gas-flow (MFC) calibration step at run 220",
                             value=False)
        chart_kind = st.radio(
            "Chart type",
            ["Individuals (I-MR): every run plotted",
             "EWMA: smoothed average, catches slow drift",
             "X-bar / R: lot averages (lots of 5)"])
        seed = st.number_input("Random seed", 0, 999, 6)

    df = history(tool, n_runs, 150 if pm_on else None, pm_quality,
                 220 if mfc_on else None, -1.5, int(seed))
    x = df[resp]
    events = []
    if pm_on:
        events.append((150, f"PM (q={pm_quality:.1f})"))
    if mfc_on:
        events.append((220, "MFC step"))

    with right:
        if chart_kind.startswith("Individuals (I-MR)"):
            ch = spc.imr_chart(x)
            we = spc.western_electric(x, ch["center"], ch["sigma"])
            marks = we["index"].unique() if len(we) else []
            st.plotly_chart(chart_with_limits(
                x, ch["center"], ch["ucl"], ch["lcl"],
                f"I-chart: {LABELS[resp]}", marks, events),
                use_container_width=True)
            if len(we):
                st.error(f"{len(we)} Western Electric rule violations - "
                         f"first at run {int(we['index'].min())}")
                st.dataframe(we.groupby("rule")
                             .agg(count=("index", "size"),
                                  first_run=("index", "min"))
                             .reset_index(), use_container_width=True)
            else:
                st.success("No run-rule violations. Tool is in control.")
        elif chart_kind.startswith("EWMA"):
            ew = spc.ewma_chart(x)
            out = (ew["z"] > ew["ucl"]) | (ew["z"] < ew["lcl"])
            marks = ew["z"][out].index
            st.plotly_chart(chart_with_limits(
                ew["z"], ew["center"], ew["ucl"], ew["lcl"],
                f"EWMA (lambda=0.2): {LABELS[resp]}", None, events),
                use_container_width=True)
            if out.any():
                st.error(f"EWMA out of limits from run {int(marks[0])} - "
                         "slow drift detected.")
            else:
                st.success("EWMA inside limits - no drift detected.")
        else:
            sub = simulate.subgrouped(df, resp, size=5)
            ch = spc.xbar_r_chart(sub, resp, "lot")
            st.plotly_chart(chart_with_limits(
                ch["xbar"], ch["x_center"], ch["x_ucl"], ch["x_lcl"],
                f"X-bar chart (n={ch['n']}): {LABELS[resp]}", None,
                [(e[0] // 5, e[1]) for e in events]),
                use_container_width=True)
            st.plotly_chart(chart_with_limits(
                ch["r"], ch["r_center"], ch["r_ucl"], ch["r_lcl"],
                "R chart"), use_container_width=True)

        with st.expander("Hidden tool state (the 'ground truth' SPC never "
                         "sees on a real tool)"):
            st.line_chart(df[["rf_match", "wall_factor"]])


# ================================================================== DOE tab
with tab_doe:
    with st.expander("What am I looking at? (plain-language guide)",
                     expanded=False):
        st.markdown("""
This tab answers the development question: *which recipe settings actually
work?* Instead of changing one knob at a time, a **DOE — Design of
Experiments** — runs a small, deliberately chosen set of recipe combinations
and fits a statistical model to the results.

- **CCD — Central Composite Design.** A standard experiment plan (corner
  points + center points + one-knob-out "star" points, 28 runs here) that is
  just rich enough to fit curved response models.
- **Response surface.** The fitted model's prediction of one output (color
  map) across two chosen knobs, with the other knobs held at their middle
  values.
- **Process window.** You type in specification limits (min/max) for every
  output; the **green boundary** encloses the region where *all* of them are
  met simultaneously. Real process development is exactly this: finding and
  centering in that window.
- **Effects table** (expander at the bottom). Which knobs move which output,
  how strongly, and how confidently (p-value). Coded units mean every knob is
  scaled to -1...+1, so the numbers are directly comparable.

**Try:** tighten one spec and watch the green window shrink. If the window
disappears, the specs are not achievable with this tool — also an answer.
""")
    left, right = st.columns([1, 3])
    with left:
        tool_name2 = st.selectbox("Tool", list(TOOLS), key="doe_tool",
                                  index=1)
        tool2 = TOOLS[tool_name2]
        df2, factors = run_ccd(tool2, 11)
        names = list(factors)
        resps = RESPONSES[tool2]
        st.markdown(f"Face-centred CCD, **{len(df2)} runs**, quadratic "
                    "response-surface models.")
        xf = st.selectbox("X axis", names, index=0)
        y_opts = [n for n in names if n != xf]
        yf = st.selectbox("Y axis", y_opts, index=len(y_opts) - 1)
        resp2 = st.selectbox("Response surface", resps,
                             format_func=lambda r: LABELS[r])
        st.markdown("**Spec limits** define the green process window "
                    "(leave a field empty for no limit):")
        DEFAULT_SPECS = {"etch_rate_nm_min": ("60", ""),
                         "dep_rate_nm_min": ("5", ""),
                         "nonuniformity_pct": ("", "2.6"),
                         "selectivity": ("12", ""),
                         "defect_density_au": ("", "3"),
                         "resistivity_mohm_cm": ("", "1.2"),
                         "transmittance_pct": ("85", "")}

        def parse(s):
            s = s.strip()
            try:
                return float(s) if s else None
            except ValueError:
                return None

        specs = {}
        for r in resps:
            lo_def, hi_def = DEFAULT_SPECS.get(r, ("", ""))
            st.caption(LABELS[r])
            c_lo, c_hi = st.columns(2)
            lo_s = c_lo.text_input("min", value=lo_def, key=f"lo_{r}",
                                   label_visibility="collapsed",
                                   placeholder="min")
            hi_s = c_hi.text_input("max", value=hi_def, key=f"hi_{r}",
                                   label_visibility="collapsed",
                                   placeholder="max")
            specs[r] = (parse(lo_s), parse(hi_s))

    fits = {r: doe.fit_response(df2, r, names, quadratic=True)
            for r in resps}
    with right:
        g1, g2, z = doe.response_surface(fits[resp2], names, xf, yf)
        _, _, ok = doe.process_window(fits, names, xf, yf, specs)

        lo_x, hi_x = factors[xf]
        lo_y, hi_y = factors[yf]
        real_x = lo_x + (g1[0] + 1) * (hi_x - lo_x) / 2
        real_y = lo_y + (g2[:, 0] + 1) * (hi_y - lo_y) / 2

        fig = go.Figure()
        fig.add_contour(x=real_x, y=real_y, z=z,
                        colorscale="Viridis",
                        colorbar=dict(title=LABELS[resp2]))
        fig.add_contour(x=real_x, y=real_y, z=ok.astype(int),
                        showscale=False, opacity=0.35,
                        contours=dict(start=0.5, end=0.5, coloring="lines"),
                        line=dict(color="lime", width=3),
                        name="window edge")
        fig.update_layout(title=f"{LABELS[resp2]} over {xf} x {yf} "
                                f"(others at centre) - window edge in green",
                          xaxis_title=xf, yaxis_title=yf, height=520,
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Share of this slice inside all specs",
                  f"{ok.mean() * 100:.0f}%")

        with st.expander("Effects tables (coded units) and model quality"):
            for r in resps:
                st.markdown(f"**{LABELS[r]}** - R2 = "
                            f"{fits[r].rsquared:.2f}")
                st.dataframe(doe.effects_table(fits[r]).head(8).round(3),
                             use_container_width=True)


# ====================================================================== APC
@st.cache_data(show_spinner="Running the controlled scenario...")
def apc_pair(n_runs, pm_quality, lam, deadband_sigma, seed):
    from plasmafab import apc as apc_mod
    ctl = apc_mod.controlled_history(n_runs=n_runs, pm_at=150,
                                     pm_quality=pm_quality, lam=lam,
                                     deadband_sigma=deadband_sigma,
                                     seed=int(seed))
    unc = simulate.production_history("pecvd", n_runs=n_runs, pm_at=150,
                                      pm_quality=pm_quality, seed=int(seed))
    return ctl, unc


with tab_apc:
    with st.expander("What am I looking at? (plain-language guide)",
                     expanded=False):
        st.markdown("""
The monitoring tab only *detects* problems. This tab *fixes* them
automatically: **APC — Advanced Process Control**, here in its most common
form, the **run-to-run controller** (adjusts the recipe between runs, not
during a run).

The logic after every run: compare the measurement to the target, update a
smoothed estimate of how far off the tool currently is (the same EWMA
averaging as on the monitoring tab), and trim one recipe knob — here RF
power — so the *next* run lands back on target.

- **Grey trace:** the tool left alone. After the bad maintenance at run 150
  it drifts off target and stays off.
- **Dark trace:** the same tool, same fault, with the controller active. The
  bottom chart shows the price: RF power quietly climbs to compensate.
- **Lambda:** the controller's memory. High = reacts fast but chases random
  noise; low = smooth but slow to respond to a real shift. Try both extremes.
- **Deadband:** "don't touch the knob while the error is smaller than X" —
  avoids constant fiddling driven by pure noise.
- **Cpk** (see Capability tab) improves because the process is re-centered.
  It does not become perfect: feedback can restore the *average*, but it
  cannot remove the run-to-run *scatter* of the tool itself.
""")
    left, right = st.columns([1, 3])
    with left:
        st.markdown("**Monitoring detects, control compensates.** An EWMA "
                    "run-to-run controller trims RF power after every run "
                    "to hold the deposition rate on target through tool "
                    "aging and a bad maintenance at run 150.")
        n_runs4 = st.slider("Production runs", 200, 600, 300, 50,
                            key="apc_runs")
        pm_q4 = st.slider("PM quality", 0.0, 1.0, 0.3, 0.1, key="apc_pmq")
        lam4 = st.slider("Controller lambda", 0.1, 0.6, 0.3, 0.05,
                         help="EWMA forgetting factor: higher reacts "
                              "faster but chases noise more.")
        db4 = st.slider("Deadband (in sigma)", 0.0, 1.5, 0.5, 0.25,
                        help="No correction while the predicted error is "
                             "inside this band.")
        seed4 = st.number_input("Random seed", 0, 999, 2, key="apc_seed")

    ctl, unc = apc_pair(n_runs4, pm_q4, lam4, db4, seed4)
    tgt = ctl.attrs["target"]
    lsl4, usl4 = tgt * 0.97, tgt * 1.03
    cpk_u = spc.capability(unc["dep_rate_nm_min"].loc[150:], lsl4, usl4)["cpk"]
    cpk_c = spc.capability(ctl["dep_rate_nm_min"].loc[150:], lsl4, usl4)["cpk"]

    with right:
        fig = go.Figure()
        fig.add_scatter(x=unc.index, y=unc["dep_rate_nm_min"], mode="lines",
                        name="open loop (SPC only)",
                        line=dict(color="#b0b7bf", width=1))
        fig.add_scatter(x=ctl.index, y=ctl["dep_rate_nm_min"], mode="lines",
                        name="R2R controlled",
                        line=dict(color="#1b3a5b", width=1.2))
        fig.add_hline(y=tgt, line_dash="dot", line_color="green",
                      annotation_text="target")
        fig.add_hrect(y0=lsl4, y1=usl4, fillcolor="green", opacity=0.07,
                      line_width=0)
        fig.add_vline(x=150, line_dash="dashdot", line_color="orange",
                      annotation_text=f"PM (q={pm_q4:.1f})")
        fig.update_layout(title="Deposition rate: open loop vs run-to-run "
                                "controlled", height=340,
                          margin=dict(l=10, r=10, t=40, b=10),
                          legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_scatter(x=ctl.index, y=ctl["rf_power_w"], mode="lines",
                         name="RF power", line=dict(color="#c05621",
                                                    width=1.2))
        fig2.add_vline(x=150, line_dash="dashdot", line_color="orange")
        fig2.update_layout(title="Controller action: RF power absorbs aging "
                                 "and the PM step", height=240,
                           margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("post-PM Cpk, open loop", f"{cpk_u:.2f}")
        m2.metric("post-PM Cpk, controlled", f"{cpk_c:.2f}")
        m3.metric("clamped runs", int(ctl["clamped"].sum()))
        st.caption("Honest limit: R2R restores centering; it cannot reduce "
                   "the tool's inherent run-to-run noise. Controller gain "
                   "comes from the DOE response-surface slope.")


# =============================================================== capability
with tab_cap:
    with st.expander("What am I looking at? (plain-language guide)",
                     expanded=False):
        st.markdown("""
**Process capability** asks: how comfortably does the process fit inside the
specification limits a customer (or the next process step) demands?

- **LSL / USL — Lower / Upper Specification Limit.** The tolerance band the
  product must stay inside. Note these come from the *product*, unlike
  control limits, which come from the *process*.
- **Cp.** Ratio of the spec width to the process spread (6 standard
  deviations). Cp = 1 means the distribution just barely fits.
- **Cpk.** Same idea, but also punishes being off-center: it measures the
  distance from the process mean to the *nearest* spec limit. Cpk is always
  ≤ Cp.
- **Rules of thumb:** Cpk ≥ 1.33 counts as a capable process; ≥ 1.67 is
  expected for critical parameters.

**Try:** drag the limits closer together and watch Cp/Cpk fall — that's the
negotiation between what the design wants and what the tool can do.
""")
    st.markdown("Process capability of the qualified recipe on a healthy "
                "tool (no injected faults).")
    tool3 = TOOLS[st.selectbox("Tool", list(TOOLS), key="cap_tool")]
    resp3 = st.selectbox("Response", RESPONSES[tool3],
                         format_func=lambda r: LABELS[r], key="cap_resp")
    dfh = history(tool3, 300, 150, 1.0, None, 0.0, 3)
    xh = dfh[resp3]
    c1, c2 = st.columns(2)
    lsl = c1.number_input("LSL", value=float(round(xh.mean()
                                                   - 4 * xh.std(), 2)))
    usl = c2.number_input("USL", value=float(round(xh.mean()
                                                   + 4 * xh.std(), 2)))
    cap = spc.capability(xh, lsl, usl)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("mean", f"{cap['mean']:.3f}")
    m2.metric("sigma", f"{cap['sigma']:.3f}")
    m3.metric("Cp", f"{cap['cp']:.2f}" if cap["cp"] else "n/a")
    m4.metric("Cpk", f"{cap['cpk']:.2f}" if cap["cpk"] else "n/a")

    fig = go.Figure()
    fig.add_histogram(x=xh, nbinsx=40, name="distribution")
    fig.add_vline(x=lsl, line_color="red", annotation_text="LSL")
    fig.add_vline(x=usl, line_color="red", annotation_text="USL")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                      title=LABELS[resp3])
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Rule of thumb: Cpk >= 1.33 is a capable process; "
               "Cpk >= 1.67 for critical parameters.")
