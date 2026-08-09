"""plasmafab: a virtual plasma fab for DOE, SPC and APC practice.

Modules:
  models    physics-light PECVD / RIE / sputter tool models with drift
  doe       factorial / CCD designs, OLS effects, response surfaces, windows
  spc       I-MR, X-bar/R, EWMA charts, Cp/Cpk, Western Electric rules
  apc       EWMA run-to-run controller (the feedback layer above SPC)
  simulate  production run histories with injectable faults
"""
from . import apc, doe, models, simulate, spc  # noqa: F401

__version__ = "0.1.0"
