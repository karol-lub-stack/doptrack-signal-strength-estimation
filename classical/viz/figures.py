import matplotlib.pyplot as plt
import numpy as np


def plot_spectrogram(pwr, t_ax, f_ax, alo, ahi, ax=None, cmap="turbo",
                     db_floor_pct=60.0, db_ceil_pct=99.5,
                     f_path=None, t_path=None):
    """Waterfall spectrogram: time on x-axis, frequency on y-axis.

    pwr : (N, n_frames) linear power. Plotted in dB with percentile colour limits.
    f_path, t_path : optional Doppler track to overlay (baseband Hz vs s).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    pwr_db = 10 * np.log10(pwr + np.finfo(float).tiny)
    finite = np.isfinite(pwr_db)
    vmin = np.percentile(pwr_db[finite], db_floor_pct)
    vmax = np.percentile(pwr_db[finite], db_ceil_pct)

    im = ax.imshow(pwr_db, cmap=cmap, aspect="auto", origin="lower",
                   vmin=vmin, vmax=vmax,
                   extent=[t_ax.min(), t_ax.max(), f_ax.min(), f_ax.max()])

    plt.colorbar(im, ax=ax, label="power (dB)")
    if f_path is not None and t_path is not None:
        ax.plot(t_path, f_path, color="red", lw=1.0, label="Doppler track")
        ax.plot(t_path, f_path + alo, color="white", lw=1.0, ls="--")
        ax.plot(t_path, f_path + ahi, color="white", lw=1.0, ls="--")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    ax.set_title("Spectrogram")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    return ax


def plot_snr(snr_db, t_ax, snr_sm2=None, ax=None):
    """Raw and smoothed SNR over time."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    ax.plot(t_ax, snr_db, color="lightgray", alpha=0.6, label="raw SNR")
    if snr_sm2 is not None:
        ax.plot(t_ax, snr_sm2, color="red", linewidth=2, label="smoothed SNR")

    if np.any(np.isfinite(snr_db)):
        ax.set_ylim(np.nanmin(snr_db) - 3, np.nanmax(snr_db) + 3)
    ax.axhline(0.0, color="0.4", lw=0.7, ls=":")
    ax.set_title("Signal strength estimation")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("SNR (dB)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def make_pass_figure(result, sat_id, save_path=None, show=False):
    """Save spectrogram and SNR as two separate figures.

    save_path should end in .png; the SNR figure is saved as <name>_snr.png.
    """
    name = result.get("name", "")

    # --- spectrogram ---
    fig_spec, ax1 = plt.subplots(figsize=(12, 6))
    plot_spectrogram(result["pwr"], result["t_ax"], result["f_ax"],
                     result["bwlo"], result["bwhi"], ax=ax1,
                     f_path=result["f_interp"], t_path=result["t_interp"])
    fig_spec.suptitle(name, fontsize=12)
    fig_spec.tight_layout()

    # --- SNR ---
    fig_snr, ax2 = plt.subplots(figsize=(12, 4))
    plot_snr(result["snr_db"], result["t_ax"], snr_sm2=result["snr_sm2"], ax=ax2)
    fig_snr.suptitle(name, fontsize=12)
    fig_snr.tight_layout()

    if save_path:
        import os
        base, ext = os.path.splitext(save_path)
        fig_spec.savefig(save_path,            dpi=130, bbox_inches="tight")
        fig_snr.savefig(base + "_snr" + ext,   dpi=130, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig_spec)
        plt.close(fig_snr)

    return fig_spec, fig_snr
