import numpy as np

LN2 = np.log(2.0)


def signal_noise_power(S):
    """Linear power matrix and per-frame noise floor (N0).

    S    : (N, n_frames) complex STFT.
    Returns
      power : (N, n_frames) linear power |S|**2.
      n0    : (n_frames,) per-frame noise floor = median over frequency bins,
              debiased by /ln2 (single-look spectrogram bins are exponential,
              so median = N0*ln2 -> divide by ln2 to recover the mean N0).
    """
    power = np.abs(S) ** 2                       # (N, n_frames)
    n0 = np.median(power, axis=0) / LN2          # (n_frames,) median over freq
    return power, n0
