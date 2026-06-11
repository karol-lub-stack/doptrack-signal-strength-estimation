"""Single-pass entry point.

process_pass() runs the full pipeline for one recording and returns a result
dict (no plotting, no I/O side effects beyond reading). Run this file directly
to process one pass and show the combined spectrogram + SNR figure; Batch.py
imports process_pass to run many passes.
"""

import numpy as np

from .io import read_data, read_tuning_freq
from .stft import stft
from .interpolation import interp
from .snr import strest
from .smoothing import smooth
from .bandwidth import get_bandwidth


def process_pass(path_iq, path_dat, f_tune,
                 f_s=25_000, N=2 ** 12, overlap_frac=0.5,
                 fixed_bw=1200, bw_mode="fixed", max_halfband_hz=2500.0,
                 smoother="butterworth", gap_threshold=7.0,
                 count=-1, name="", also_estimate_bw=False):
    """Run the pipeline for one pass.

    bw_mode : 'fixed'    -> integrate over +/- fixed_bw/2 around the carrier.
              'estimate' -> estimate the band from the data (fall back to fixed
                            if estimation fails).
    also_estimate_bw : if True, always compute the estimated band (for testing /
              comparison) and store it, even when bw_mode='fixed'.
    smoother : 'savgol', 'butterworth', or None  (see smoothing.smooth).

    Returns a dict with everything needed for plotting and summary.
    """
    N = int(N)
    hop = int(N * (1 - overlap_frac))

    loaded = read_data(path_iq, path_dat, dtype=np.complex64, count=-1)
    if loaded is None:
        raise RuntimeError(f"failed to read {path_iq} / {path_dat}")
    sig, t_bf, f_bf = loaded

    t_ax, f_ax, S = stft(sig, N, hop, f_s, window=np.hanning(N))
    pwr = np.abs(S) ** 2

    # baseband Doppler curve interpolated onto the STFT time grid
    f_bf_bb = f_bf - f_tune
    t_interp, f_interp = interp(t_ax, t_bf, f_bf_bb, gap_threshold=gap_threshold)

    # bandwidth: fixed or estimated
    bw_estimated = None
    if bw_mode == "estimate" or also_estimate_bw:
        bw_estimated = get_bandwidth(pwr, f_ax, f_interp, max_halfband_hz)

    if bw_mode == "estimate" and bw_estimated is not None:
        bw_used = bw_estimated
    else:
        bw_used = fixed_bw                                # symmetric +/- fixed_bw/2

    # SNR along the path — noise floor estimated per frame from out-of-band bins
    snr_lin, snr_db = strest(pwr, f_ax, f_interp, bw=bw_used)
    snr_lin2, snr_db2 = strest(pwr, f_ax, f_interp, bw=bw_estimated)

    # smooth
    snr_sm = smooth(snr_db, t_ax, method=smoother)
    snr_sm2 = smooth(snr_db2, t_ax, method=smoother)

    bw_used_hz = (bw_used[1] - bw_used[0]) if isinstance(bw_used, (tuple, list)) else float(bw_used)
    bw_est_hz = (bw_estimated[1] - bw_estimated[0]) if bw_estimated is not None else np.nan

    return dict(
        name=name, f_tune=f_tune, f_s=f_s, N=N, hop=hop,
        t_ax=t_ax, f_ax=f_ax, pwr=pwr,
        t_bf=t_bf, f_bf_bb=f_bf_bb,
        t_interp=t_interp, f_interp=f_interp,
        snr_lin=snr_lin, snr_db=snr_db,
        snr_lin2=snr_lin2, snr_db2=snr_db2, snr_sm=snr_sm, snr_sm2=snr_sm2,
        bw_mode=bw_mode, bw_used=bw_used, bw_used_hz=bw_used_hz,
        bw_estimated=bw_estimated, bw_estimated_hz=bw_est_hz,
        bwlo=bw_estimated[0] if bw_estimated is not None else np.nan,
        bwhi=bw_estimated[1] if bw_estimated is not None else np.nan,
        gap_threshold=gap_threshold,
    )


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from viz.figures import make_pass_figure

    # --- configure one pass ---
    base = r"X:\data"
    aid = r"\Delfi-C3_32789_202003231040"
    path_iq = base + r"\L0\Delfi-C3\2020" + aid + ".fc32"
    path_dat = base + r"\L1B\Delfi-C3\2020" + aid + ".dat"
    path_yml = base + r"\L0\Delfi-C3\2020" + aid + ".yml"

    try:
        f_tune = read_tuning_freq(path_yml)
    except Exception as e:
        print(f"yml read failed ({e}); using hardcoded f_tune")
        f_tune = 145_869_000

    result = process_pass(
        path_iq, path_dat, f_tune,
        f_s=25_000, N=2 ** 12, overlap_frac=0.5,
        fixed_bw=1200, bw_mode="fixed",
        smoother="butterworth",
        name=aid.strip("\\"),
        also_estimate_bw=True,
    )

    print(f"bw used      : {result['bw_used_hz']:.0f} Hz ({result['bw_mode']})")
    if result["bw_estimated"] is not None:
        print(f"bw estimated : {result['bw_estimated_hz']:.0f} Hz")
    if np.any(np.isfinite(result["snr_db"])):
        print(f"peak SNR     : {np.nanmax(result['snr_db']):.1f} dB")
        print(f"median SNR   : {np.nanmedian(result['snr_db']):.1f} dB")

    make_pass_figure(result, show=True)
