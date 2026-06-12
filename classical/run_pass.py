"""Run the pipeline for a single pass and display the combined figure.

Edit the config block below, then:  python run_pass.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from pipeline.io import read_tuning_freq
from pipeline.process_pass import process_pass
from viz.figures import make_pass_figure

# --------------------------------------------------------------------------- #
# config — edit these
# --------------------------------------------------------------------------- #
BASE_IQ  = r"X:\data\L0"
BASE_DAT = r"X:\data\L1B"

SATELLITE = "FUNcube-1"
YEAR      = "2026"
PASS_ID   = "FUNcube-1_39444_202601010247"

# pipeline params (keep in sync with Batch.py)
F_S          = 25_000
N            = 2 ** 12
OVERLAP_FRAC = 0.5
FIXED_BW     = 1200
BW_MODE      = "fixed"
SMOOTHER     = "butterworth"
GAP_THRESHOLD = 7.0
# --------------------------------------------------------------------------- #

path_iq  = os.path.join(BASE_IQ,  SATELLITE, YEAR, PASS_ID + ".fc32")
path_dat = os.path.join(BASE_DAT, SATELLITE, YEAR, PASS_ID + ".dat")
path_yml = os.path.join(BASE_IQ,  SATELLITE, YEAR, PASS_ID + ".yml")

try:
    f_tune = read_tuning_freq(path_yml)
    print(f"f_tune: {f_tune:.0f} Hz")
except Exception as e:
    sys.exit(f"Could not read tuning frequency from yml: {e}")

result = process_pass(
    path_iq, path_dat, f_tune,
    f_s=F_S, N=N, overlap_frac=OVERLAP_FRAC,
    fixed_bw=FIXED_BW, bw_mode=BW_MODE,
    smoother=SMOOTHER, gap_threshold=GAP_THRESHOLD,
    name=PASS_ID, also_estimate_bw=True,
)

print(f"bw used      : {result['bw_used_hz']:.0f} Hz ({result['bw_mode']})")
if result["bw_estimated"] is not None:
    print(f"bw estimated : {result['bw_estimated_hz']:.0f} Hz")
if np.any(np.isfinite(result["snr_db"])):
    print(f"peak SNR     : {np.nanmax(result['snr_db']):.1f} dB")
    print(f"median SNR   : {np.nanmedian(result['snr_db']):.1f} dB")

make_pass_figure(result, sat_id=SATELLITE, show=True)
