"""Optional data-driven bandwidth estimation.

The pipeline integrates SNR over a band around the Doppler carrier. By default
that band is a fixed, user-supplied width (consistent and comparable across
passes). For an unknown satellite you can instead estimate the band from the
data with get_bandwidth(), which:

  1. picks the strongest frames of the pass (highest coarse in-band SNR),
  2. carrier-aligns and averages them (Doppler is steepest where signal is
     strongest, so alignment is needed before averaging),
  3. walks outward from the peak to find the band edges.

Returns signed offsets (offset_lo, offset_hi) relative to the carrier, suitable
to pass straight into strest as bw=(offset_lo, offset_hi).
"""

from __future__ import annotations
import numpy as np

LN2 = np.log(2.0)


def _walk_edge(psd, center_idx, step, thr_low, debounce, max_bins):
    """Extend an edge outward from center while PSD stays above thr_low.
    A single sub-threshold bin does not stop the walk; `debounce` consecutive
    sub-threshold bins do (ragged-edge tolerance). Capped at max_bins.
    """
    n, edge, below, moved, i = psd.size, center_idx, 0, 0, center_idx
    while 0 <= i + step < n and moved < max_bins:
        i += step
        moved += 1
        if psd[i] >= thr_low:
            edge, below = i, 0
        else:
            below += 1
            if below >= debounce:
                break
    return edge


def select_strong_frames(power, f_axis, fc, max_halfband_hz, frac=0.15, min_frames=5):
    """Rank frames by a coarse in-band SNR proxy; return indices of the top `frac`.

    power : (N, n_frames) linear power. fc : (n_frames,) carrier per frame [Hz].
    """
    df = float(np.median(np.diff(f_axis)))
    proxy = np.full(power.shape[1], -np.inf)
    for k in range(power.shape[1]):
        if not np.isfinite(fc[k]):
            continue
        win = np.abs(f_axis - fc[k]) <= max_halfband_hz
        if not win.any():
            continue
        floor = np.median(power[~win, k]) / LN2 if (~win).any() else 0.0
        proxy[k] = np.sum(power[win, k] - floor) * df
    order = np.argsort(proxy)[::-1]
    order = order[np.isfinite(proxy[order])]
    n = max(min_frames, int(round(frac * order.size)))
    return order[:max(1, min(n, order.size))]


def estimate_band(power, f_axis, fc, frame_idx, max_halfband_hz,
                  alpha_hi_db=6.83, alpha_lo_db=1, debounce=26, min_bw_hz=1200):
    """Carrier-align the chosen frames, average, then walk the band edges.

    Returns dict(present, offset_lo, offset_hi) -- offsets are Hz relative to
    the carrier (offset_lo negative, offset_hi positive for a centred band).
    """
    df = float(np.median(np.diff(f_axis)))
    g = np.arange(-max_halfband_hz, max_halfband_hz + df, df)   # carrier-relative grid
    acc, cnt = np.zeros_like(g), 0
    for k in frame_idx:
        if not np.isfinite(fc[k]):
            continue
        acc += np.interp(g, f_axis - fc[k], power[:, k], left=np.nan, right=np.nan)
        cnt += 1
    if cnt == 0:
        raise ValueError("No usable frames for band estimation.")
    bar = acc / cnt
    valid = np.isfinite(bar)

    # averaged bins are ~gamma(cnt): median ~ mean, so a plain median floor is fine
    n0 = float(np.median(bar[valid]))
    thr_hi = n0 * 10 ** (alpha_hi_db / 10)
    thr_lo = n0 * 10 ** (alpha_lo_db / 10)

    # centre = local max near g = 0 (tolerates small carrier error)
    z = int(np.argmin(np.abs(g)))
    w = max(1, int(round((max_halfband_hz / 4) / df)))
    lo, hi = max(0, z - w), min(g.size, z + w + 1)
    c = lo + int(np.nanargmax(np.where(valid[lo:hi], bar[lo:hi], -np.inf)))

    if not (bar[c] >= thr_hi):
        return dict(present=False, offset_lo=np.nan, offset_hi=np.nan)

    nmax = int(round(max_halfband_hz / df))
    walk = np.where(valid, bar, -np.inf)
    ei = _walk_edge(walk, c, +1, thr_lo, debounce, nmax)
    ej = _walk_edge(walk, c, -1, thr_lo, debounce, nmax)
    ol = float(g[ej])
    oh = float(g[ei])
    print(ol, oh)

    return dict(present=True, offset_lo=ol, offset_hi=oh)


def get_bandwidth(power, f_axis, fc, max_halfband_hz=2500.0):
    """Public entry point.

    Returns (offset_lo, offset_hi) signed offsets [Hz] relative to the carrier,
    or None if no band could be confidently estimated. Pass the tuple straight
    to strest as bw=..., or fall back to a fixed width: bw = get_bandwidth(...) or 1200.
    """
    try:
        strong = select_strong_frames(power, f_axis, fc, max_halfband_hz)
        if strong.size == 0:
            return None
        band = estimate_band(power, f_axis, fc, strong, max_halfband_hz)
        if not band["present"]:
            return None
        return (band["offset_lo"], band["offset_hi"])
    except Exception as e:
        print(f"band estimation failed: {e}")
        return None
