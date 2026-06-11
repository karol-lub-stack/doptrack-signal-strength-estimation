import numpy as np

LN2 = np.log(2.0)


def strest(pwr, f_ax, f_path, bw=1200):
    """Per-frame in-band SNR along the Doppler path.

    Noise floor is estimated per frame from the out-of-band bins only
    (debiased median / ln2), so it tracks time-varying conditions and is
    not contaminated by signal energy inside the band.

    Parameters
    ----------
    pwr    : (N, n_frames) linear power matrix.
    f_ax   : (N,) frequency axis [Hz], ascending.
    f_path : (n_frames,) interpolated Doppler path [Hz, baseband]; NaN where undefined.
    bw     : integration bandwidth [Hz] for a symmetric band, OR a tuple
             (offset_lo, offset_hi) of signed offsets relative to the carrier
             (e.g. from bandwidth.get_bandwidth).

    Returns
    -------
    snr_linear : (n_frames,) linear SNR; NaN where the path is undefined.
    snr_db     : (n_frames,) 10*log10(SNR); NaN where SNR <= 0 or undefined.
    """
    if isinstance(bw, (tuple, list)):
        offset_lo, offset_hi = bw
    else:
        offset_lo, offset_hi = -bw / 2.0, bw / 2.0

    n_steps = len(f_path)
    snr_linear = np.full(n_steps, np.nan)

    for i in range(n_steps):
        fc = f_path[i]
        if not np.isfinite(fc):
            continue
        band = (f_ax >= fc + offset_lo) & (f_ax <= fc + offset_hi)
        if not band.any() or not (~band).any():
            continue
        n0 = float(np.median(pwr[~band, i]) / LN2)   # out-of-band noise floor, per frame
        if not np.isfinite(n0) or n0 <= 0:
            continue
        excess = pwr[band, i] - n0
        snr_linear[i] = float(np.sum(excess) / (band.sum() * n0))

    with np.errstate(invalid="ignore", divide="ignore"):
        snr_db = 10 * np.log10(np.where(snr_linear > 0, snr_linear, np.nan))

    return snr_linear, snr_db
