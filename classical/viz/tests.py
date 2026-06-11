"""Validation tests for the pipeline.

Two quantitative checks plus a combined diagnostic figure:

  check_interpolation_gaps : verifies the gap-detection in interp() -- every NaN
      run in f_interp should correspond to a gap > gap_threshold in the .dat
      timestamps, and there should be no NaNs strictly inside a valid segment.

  check_bandwidth : compares an estimated bandwidth against a known truth within
      a relative tolerance.

  make_test_figure : a separate combined figure (interpolation overlay + aligned
      band average) for visual inspection per pass.
"""

import numpy as np
import matplotlib.pyplot as plt

from pipeline.diagnostics import diagnose_alignment   # re-exported for convenience


def check_interpolation_gaps(t_ax, f_interp, t_bf, gap_threshold=7.0):
    """Check that NaN regions in the interpolated curve match real .dat gaps.

    Returns a dict: n_nan, n_valid, n_big_gaps (gaps>threshold in t_bf), and
    'ok' (bool) -- ok means every interpolated frame that falls strictly inside
    a <=threshold segment is finite, and frames inside big gaps are NaN.
    """
    t_ax = np.asarray(t_ax, float)
    f_interp = np.asarray(f_interp, float)
    t_bf = np.asarray(t_bf, float)

    nan = ~np.isfinite(f_interp)

    # expected-valid mask: inside any consecutive pair with gap <= threshold
    valid_expected = np.zeros(t_ax.size, dtype=bool)
    n_big_gaps = 0
    for i in range(1, t_bf.size):
        dt = t_bf[i] - t_bf[i - 1]
        seg = (t_ax >= t_bf[i - 1]) & (t_ax <= t_bf[i])
        if dt <= gap_threshold:
            valid_expected[seg] = True
        else:
            n_big_gaps += 1

    # frames expected valid but NaN, or expected invalid but finite -> mismatch
    mismatch_valid_but_nan = int(np.sum(valid_expected & nan))
    mismatch_invalid_but_finite = int(np.sum((~valid_expected) & (~nan)
                                             & (t_ax >= t_bf.min())
                                             & (t_ax <= t_bf.max())))
    ok = (mismatch_valid_but_nan == 0 and mismatch_invalid_but_finite == 0)

    return dict(
        ok=bool(ok),
        n_nan=int(np.sum(nan)),
        n_valid=int(np.sum(~nan)),
        n_big_gaps=n_big_gaps,
        mismatch_valid_but_nan=mismatch_valid_but_nan,
        mismatch_invalid_but_finite=mismatch_invalid_but_finite,
    )


def check_bandwidth(estimated_hz, true_hz, rel_tol=0.20):
    """Compare an estimated bandwidth to a known truth within rel_tol (fraction).

    Returns dict(ok, estimated_hz, true_hz, rel_error). ok is False if the
    estimate is NaN/None.
    """
    if estimated_hz is None or not np.isfinite(estimated_hz) or true_hz in (None, 0):
        return dict(ok=False, estimated_hz=estimated_hz, true_hz=true_hz, rel_error=np.nan)
    rel = abs(estimated_hz - true_hz) / true_hz
    return dict(ok=bool(rel <= rel_tol), estimated_hz=float(estimated_hz),
                true_hz=float(true_hz), rel_error=float(rel))


def make_test_figure(result, true_bw_hz=None, save_path=None, show=False):
    """Combined diagnostic figure: interpolation overlay (top) and the
    carrier-aligned band average used for bandwidth estimation (bottom).
    """
    t_ax = result["t_ax"]
    t_bf = result["t_bf"]
    f_bf_bb = result["f_bf_bb"]
    f_interp = result["f_interp"]

    fig, (ax1, ax2) = plt.subplots(2, figsize=(14, 9))

    # --- top: interpolation vs original .dat points ---
    ax1.plot(t_bf, f_bf_bb, "o", ms=3, color="black", label=".dat points")
    ax1.plot(t_ax, f_interp, "-", lw=1.2, color="red", label="interpolated")
    nan = ~np.isfinite(f_interp)
    if nan.any():
        ymin = np.nanmin(f_interp)
        ax1.plot(t_ax[nan], np.full(nan.sum(), ymin), "|", color="orange",
                 ms=8, label="NaN (gap)")
    ax1.set_title("Interpolation & gap detection")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("baseband freq (Hz)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # --- bottom: aligned band average (recomputed for display) ---
    try:
        from pipeline.bandwidth import select_strong_frames, estimate_band
        pwr, f_ax = result["pwr"], result["f_ax"]
        strong = select_strong_frames(pwr, f_ax, f_interp, 2500.0)
        df = float(np.median(np.diff(f_ax)))
        g = np.arange(-2500.0, 2500.0 + df, df)
        acc, cnt = np.zeros_like(g), 0
        for k in strong:
            if not np.isfinite(f_interp[k]):
                continue
            acc += np.interp(g, f_ax - f_interp[k], pwr[:, k], left=np.nan, right=np.nan)
            cnt += 1
        bar = acc / max(cnt, 1)
        ax2.plot(g, 10 * np.log10(bar + np.finfo(float).tiny), color="C0",
                 label="aligned avg (strong frames)")
        if result["bw_estimated"] is not None:
            lo, hi = result["bw_estimated"]
            ax2.axvline(lo, color="red", ls="--", lw=1)
            ax2.axvline(hi, color="red", ls="--", lw=1, label="estimated edges")
        if true_bw_hz is not None:
            ax2.axvline(-true_bw_hz / 2, color="green", ls=":", lw=1)
            ax2.axvline(true_bw_hz / 2, color="green", ls=":", lw=1, label="true edges")
        ax2.set_title("Carrier-aligned band average")
        ax2.set_xlabel("offset from carrier (Hz)")
        ax2.set_ylabel("power (dB)")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
    except Exception as e:
        ax2.text(0.5, 0.5, f"band view unavailable: {e}", ha="center")

    fig.suptitle(result.get("name", "") + "  -- tests", fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
