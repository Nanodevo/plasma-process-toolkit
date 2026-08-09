#!/usr/bin/env python3
"""Cluster tool vs. standalone tools: why one-unit systems exist.

Deposits the full TCO / a-Si:H p-i-n / TCO stack 40 times in each mode:

  cluster     PECVD + sputter chambers on one vacuum system (the way the
              thesis samples were produced) - no air exposure between
              the TCO and silicon layers
  standalone  two separate tools with a vacuum break in between

and compares interface quality and the overall stack penalty.
Run:  python examples/run_stack.py
"""
import numpy as np
import pandas as pd

from plasmafab import simulate


def batch(vacuum_break: bool, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        r = simulate.stack_run(vacuum_break=vacuum_break, rng=rng)
        rows.append({
            "interface_defects_au": r["interface_defects_au"],
            "contact_resistivity_mohm_cm": r["contact_resistivity_mohm_cm"],
            "stack_penalty_au": r["stack_penalty_au"],
        })
    return pd.DataFrame(rows)


def main():
    cluster = batch(vacuum_break=False, n=40, seed=5)
    standalone = batch(vacuum_break=True, n=40, seed=5)

    summary = pd.DataFrame({
        "cluster (one unit)": cluster.mean(),
        "standalone (air break)": standalone.mean(),
    })
    summary["delta_%"] = (summary.iloc[:, 1] / summary.iloc[:, 0] - 1) * 100
    print("Mean over 40 stacks each:\n")
    print(summary.round(2).to_string())
    print("\nThe air break costs a factor of ~{:.1f} in interface defects - "
          "the reason the thesis samples were deposited in a one-unit "
          "PECVD + sputter system without leaving vacuum.".format(
              standalone["interface_defects_au"].mean()
              / cluster["interface_defects_au"].mean()))


if __name__ == "__main__":
    main()
