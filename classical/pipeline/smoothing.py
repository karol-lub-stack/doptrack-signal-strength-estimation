"""SNR time-series smoothers.

Two options, both NaN-safe (gaps where the Doppler path is undefined are
preserved as NaN in the output):

  'savgol'      : median filter (impulse rejection) then Savitzky-Golay
                  (preserves peak shape well). The original refined choice.
  'butterworth' : median filter then zero-phase Butterworth low-pass
                  (cleaner on flat segments). Lifted from the merged pipeline.

Use smooth(x, t, method=...) as the single entry point.
"""

import numpy as np

try:
    from scipy.signal import medfilt, butter, filtfilt, savgol_filter
    from scipy.ndimage import gaussian_filter1d
    _HAVE_SCIPY = True
except Exception:                                   # pragma: no cover
    _HAVE_SCIPY = False


def _bridge_nans(x):
    """Return (y, nan_mask): a copy with NaNs linearly interpolated across so
    the filters have a continuous series to work on, plus the original mask.
    """
    x = np.asarray(x, float).copy()
    nan = ~np.isfinite(x)
    if nan.all():
        return x, nan
    idx = np.arange(x.size)
    y = x.copy()
    y[nan] = np.interp(idx[nan], idx[~nan], x[~nan])
    return y, nan


def smooth_savgol(x, median_kernel=7, window_length=101, polyorder=3):
    """Median filter then Savitzky-Golay. NaN-safe."""
    if not _HAVE_SCIPY:
        return np.asarray(x, float)
    y, nan = _bridge_nans(x)
    if nan.all():
        return y
    if median_kernel and median_kernel >= 3:
        k = median_kernel + (median_kernel % 2 == 0)        # force odd
        y = medfilt(y, kernel_size=k)
    # window_length must be odd and <= series length
    wl = min(window_length, y.size - (y.size + 1) % 2)
    if wl >= polyorder + 2:
        wl += (wl % 2 == 0)
        y = savgol_filter(y, window_length=wl, polyorder=polyorder)
    y[nan] = np.nan
    return y


def postfilter(x, t, median_kernel=7, cutoff_hz=0.1):
    """Median filter then zero-phase Butterworth low-pass. NaN-safe.

    cutoff_hz is in cycles per second of the SNR time series (so it depends on
    the STFT hop spacing, read from t). Falls back to a Gaussian smoother for
    very short series or filter-design failures.
    """
    if not _HAVE_SCIPY:
        return np.asarray(x, float)
    y, nan = _bridge_nans(x)
    if nan.all():
        return y
    if median_kernel and median_kernel >= 3:
        k = median_kernel + (median_kernel % 2 == 0)
        y = medfilt(y, kernel_size=k)
    if cutoff_hz is not None:
        fs = 1.0 / float(np.median(np.diff(t)))
        wn = float(np.clip(cutoff_hz / (fs / 2.0), 1e-4, 0.99))
        try:
            b, a = butter(2, wn)
            y = filtfilt(b, a, y) if y.size > 12 else gaussian_filter1d(y, 1.0 / wn)
        except Exception:
            y = gaussian_filter1d(y, max(1.0, 1.0 / wn))
    y[nan] = np.nan
    return y


def smooth(x, t, method="butterworth", **kwargs):
    """Dispatch to the chosen smoother.

    method='savgol'      -> smooth_savgol (median + Savitzky-Golay)
    method='butterworth' -> postfilter   (median + zero-phase Butterworth)
    method=None / 'none' -> returned unchanged
    """
    if method in (None, "none"):
        return np.asarray(x, float)
    if method == "savgol":
        return smooth_savgol(x, **kwargs)
    if method in ("butterworth", "butter", "postfilter"):
        return postfilter(x, t, **kwargs)
    raise ValueError(f"unknown smoothing method {method!r}")
