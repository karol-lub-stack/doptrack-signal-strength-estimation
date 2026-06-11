import numpy as np
from scipy.interpolate import interp1d


def interp(time_axis, bf_time, bf_freq, gap_threshold=7.0):
    """Interpolate the best-fit Doppler curve onto the STFT time grid.

    Frequencies outside a valid pass segment are set to NaN. A segment between
    two consecutive .dat points is "valid" only if their time gap is
    <= gap_threshold seconds; larger gaps are treated as missing data.

    Parameters
    ----------
    time_axis     : (n_frames,) STFT frame times [s].
    bf_time       : best-fit Doppler timestamps [s].
    bf_freq       : best-fit Doppler frequencies [Hz] (baseband if already shifted).
    gap_threshold : max gap in the .dat data, in seconds, before a segment is invalid.

    Returns
    -------
    time_axis    : unchanged (returned for convenience).
    target_freqs : (n_frames,) interpolated frequencies; NaN outside valid segments.
    """
    f_map = interp1d(bf_time, bf_freq, kind="linear", fill_value="extrapolate")
    target_freqs = f_map(time_axis)

    valid_mask = np.zeros(len(time_axis), dtype=bool)
    for i in range(1, len(bf_time)):
        dt = bf_time[i] - bf_time[i - 1]
        if dt <= gap_threshold:
            segment = (time_axis >= bf_time[i - 1]) & (time_axis <= bf_time[i])
            valid_mask[segment] = True

    target_freqs[~valid_mask] = np.nan
    return time_axis, target_freqs
