"""
Short-Time Fourier Transform for the satellite strength pipeline.

    t_stft, f_stft, Zxx = stft(x, nfft, hop, fs)

Parameters
----------
x     : 1-D signal, real or complex. Complex (IQ) baseband is expected for the
        satellite case, which yields a two-sided spectrum centred on 0 Hz.
nfft  : frame size = window length = FFT length, in samples.
hop   : samples advanced between consecutive frames -- frame i starts at i*hop,
        so the overlap is (nfft - hop) samples.
fs    : sampling rate, Hz.

Returns
-------
t_stft : (n_frames,) frame times in seconds, at the CENTRE of each window.
f_stft : (nfft,) frequencies in Hz, ascending and fftshifted: -fs/2 .. +fs/2-df.
Zxx    : (nfft, n_frames) complex STFT, indexed [frequency, frame].

Notes
-----
* scaling='density' makes |Zxx|**2 a PSD estimate (divides by fs*sum(w**2)).
  The downstream pipeline only uses ratios and noise-subtracted differences, so
  the choice of scaling shifts N0 but does not change SNR.
* The window centre is used as each frame's timestamp so the frame's energy is
  associated with the middle of its support, aligning cleanly with a Doppler
  track sampled on its own time grid.
"""

from __future__ import annotations
import numpy as np


def _get_window(window, nfft: int) -> np.ndarray:
    """Resolve a window argument to a length-nfft float array."""
    if window is None:
        return np.ones(nfft)
    if isinstance(window, np.ndarray):
        if window.size != nfft:
            raise ValueError(f"window length {window.size} != nfft {nfft}")
        return window.astype(float)
    name = str(window).lower()
    if name in ("hann", "hanning"):
        return np.hanning(nfft)
    if name == "hamming":
        return np.hamming(nfft)
    if name == "blackman":
        return np.blackman(nfft)
    if name in ("rect", "rectangular", "boxcar", "none"):
        return np.ones(nfft)
    raise ValueError(f"unknown window {window!r}")


def stft(x, nfft, hop, fs, window="hann", scaling="density", center=True):
    """Compute the STFT. See module docstring for the full contract."""
    x = np.asarray(x)
    nfft, hop = int(nfft), int(hop)
    if x.ndim != 1:
        raise ValueError("x must be 1-D")
    if nfft < 1 or hop < 1:
        raise ValueError("nfft and hop must be >= 1")
    if x.size < nfft:
        raise ValueError(f"signal length {x.size} < nfft {nfft}")

    w = _get_window(window, nfft)

    # Build frames (n_frames, nfft) as a strided view, keep every hop-th, window.
    view = np.lib.stride_tricks.sliding_window_view(x, nfft)   # (N-nfft+1, nfft)
    frames = view[::hop] * w                                   # (n_frames, nfft)
    n_frames = frames.shape[0]

    if scaling == "density":
        norm = 1.0 / np.sqrt(fs * np.sum(w ** 2))   # |Zxx|**2 ~ PSD
    elif scaling == "spectrum":
        norm = 1.0 / np.sum(w)                       # |Zxx| ~ amplitude
    elif scaling is None:
        norm = 1.0
    else:
        raise ValueError("scaling must be 'density', 'spectrum', or None")

    # FFT along the sample axis, shift zero-frequency to the centre, scale.
    Z = np.fft.fftshift(np.fft.fft(frames, n=nfft, axis=1), axes=1) * norm
    Zxx = np.ascontiguousarray(Z.T)                  # -> (nfft, n_frames)

    f_stft = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))   # ascending Hz
    starts = np.arange(n_frames) * hop
    offset = (nfft - 1) / 2.0 if center else 0.0
    t_stft = (starts + offset) / fs

    return t_stft, f_stft, Zxx
