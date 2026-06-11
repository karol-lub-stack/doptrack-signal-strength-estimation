"""
Digitize SNR curves from the SST figures.
Detects the plot area, maps pixel colours to the three curves (STFT/SST/MSST),
and reports median SNR during signal presence.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FIGURES = {
    "FUNcube-1": r"C:\Users\glute\Desktop\Project Python Folder\current\PaperCurrent\Figures\cn_snr_FUNcube-1_39444_202601010247.png",
    "Delfi-C3":  r"C:\Users\glute\Desktop\Project Python Folder\current\PaperCurrent\Figures\cn_snr_Delfi-C3_32789_202003231040.png",
}

# signal presence windows (seconds) — read from the spectrogram above
SIGNAL_WINDOWS = {
    "FUNcube-1": (50,  450),
    "Delfi-C3":  (100, 700),
}

# ── target colours for each curve (RGB, approximate) ──────────────────────────
# STFT = blue-ish, SST = bright green, MSST = dark teal/olive
CURVE_TARGETS = {
    "STFT": np.array([100, 150, 220]),   # blue
    "SST":  np.array([ 80, 200,  80]),   # green
    "MSST": np.array([ 30, 120, 100]),   # dark teal
}
COLOUR_TOL = 80   # max Euclidean distance to count as a match


def find_plot_bbox(img_arr, top_frac=0.52):
    """
    The figure has two stacked panels.  The lower (SNR) panel starts at
    roughly top_frac of the total height.  We crop to that lower panel
    and then detect the actual axes limits by finding the white background.
    """
    h, w = img_arr.shape[:2]
    panel = img_arr[int(h * top_frac):, :]

    # find rows/cols that are mostly white (axes area)
    white = np.all(panel > 230, axis=2)
    row_white = white.mean(axis=1)
    col_white = white.mean(axis=0)

    # axes bounding box = first/last row/col with >50% white neighbours
    rows = np.where(row_white > 0.5)[0]
    cols = np.where(col_white > 0.5)[0]
    if len(rows) == 0 or len(cols) == 0:
        raise RuntimeError("Could not detect axes bounding box")

    r0, r1 = rows[0],  rows[-1]
    c0, c1 = cols[0],  cols[-1]
    return panel, r0, r1, c0, c1


def extract_curve(panel, r0, r1, c0, c1, target_rgb):
    """
    For each column in the axes region, find the row whose colour is
    closest to target_rgb.  Returns (x_px, y_px) arrays.
    """
    region = panel[r0:r1, c0:c1].astype(float)
    diff   = np.linalg.norm(region - target_rgb, axis=2)   # (rows, cols)

    best_row = np.argmin(diff, axis=0)   # one row per column
    best_val = diff[best_row, np.arange(diff.shape[1])]

    mask = best_val < COLOUR_TOL
    x_px = np.where(mask)[0]
    y_px = best_row[mask]
    return x_px, y_px


def px_to_data(x_px, y_px, c0, c1, r0, r1, t_range, snr_range):
    """Map pixel coordinates to (time_s, snr_dB)."""
    t   = t_range[0]   + (x_px / (c1 - c0)) * (t_range[1]   - t_range[0])
    snr = snr_range[1] - (y_px / (r1 - r0)) * (snr_range[1] - snr_range[0])
    return t, snr


def read_axis_limits(img_arr, top_frac=0.52):
    """
    Read axis limits from figure labels.
    Hardcoded from visual inspection since OCR is out of scope:
      time axis:  0 – 700 s  (FUNcube ends ~650, Delfi ~720 — use 700)
      SNR axis:  -200 – +200 dB  (both figures)
    """
    return (0, 700), (-200, 200)


# ──────────────────────────────────────────────────────────────────────────────

results = {}

for sat, path in FIGURES.items():
    print(f"\n{'='*55}")
    print(f"  {sat}")
    print(f"{'='*55}")

    img  = Image.open(path).convert("RGB")
    arr  = np.array(img)

    try:
        panel, r0, r1, c0, c1 = find_plot_bbox(arr)
    except RuntimeError as e:
        print(f"  bbox detection failed: {e} — using fallback")
        h, w = arr.shape[:2]
        panel = arr[int(h * 0.52):, :]
        r0, r1 = int(panel.shape[0]*0.05), int(panel.shape[0]*0.88)
        c0, c1 = int(panel.shape[1]*0.08), int(panel.shape[1]*0.95)

    t_range, snr_range = read_axis_limits(arr)

    sat_results = {}
    fig, ax = plt.subplots(figsize=(10, 4))

    colours_plot = {"STFT": "steelblue", "SST": "limegreen", "MSST": "darkcyan"}
    t_win = SIGNAL_WINDOWS[sat]

    for name, target in CURVE_TARGETS.items():
        x_px, y_px = extract_curve(panel, r0, r1, c0, c1, target)
        if len(x_px) < 10:
            print(f"  {name:6s}: too few pixels matched ({len(x_px)}), skipping")
            continue

        t, snr = px_to_data(x_px, y_px, c0, c1, r0, r1, t_range, snr_range)

        # restrict to signal window
        in_win = (t >= t_win[0]) & (t <= t_win[1])
        t_sig  = t[in_win];   snr_sig = snr[in_win]

        med  = float(np.median(snr_sig))   if len(snr_sig) else np.nan
        mn   = float(np.mean(snr_sig))     if len(snr_sig) else np.nan
        peak = float(np.max(snr_sig))      if len(snr_sig) else np.nan

        sat_results[name] = dict(median=med, mean=mn, peak=peak,
                                 t=t, snr=snr, t_sig=t_sig, snr_sig=snr_sig)

        print(f"  {name:6s}  signal window {t_win[0]}–{t_win[1]} s")
        print(f"           median = {med:8.1f} dB")
        print(f"           mean   = {mn:8.1f} dB")
        print(f"           peak   = {peak:8.1f} dB")
        print(f"           n pts  = {len(snr_sig)}")

        ax.plot(t, snr, color=colours_plot[name], alpha=0.5, linewidth=0.8)
        ax.plot(t_sig, snr_sig, color=colours_plot[name], linewidth=1.5,
                label=f"{name} (med {med:.1f} dB)")

    ax.axvspan(*t_win, alpha=0.08, color="yellow", label="signal window")
    ax.set_xlabel("Time (s)");  ax.set_ylabel("SNR (dB)")
    ax.set_title(f"Digitised curves — {sat}")
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)
    out_png = path.replace(".png", "_digitised.png")
    out_svg = path.replace(".png", "_digitised.svg")
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    fig.savefig(out_svg, format="svg",  bbox_inches="tight")
    plt.close(fig)
    print(f"  saved PNG: {out_png}")
    print(f"  saved SVG: {out_svg}")

    # ── CSV export ────────────────────────────────────────────────────────────
    import csv, os
    out_csv = path.replace(".png", "_digitised.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "time_s", "snr_db", "in_signal_window"])
        for name, v in sat_results.items():
            t_win_lo, t_win_hi = SIGNAL_WINDOWS[sat]
            for ti, si in zip(v["t"], v["snr"]):
                w.writerow([name, round(float(ti), 2), round(float(si), 2),
                             int(t_win_lo <= ti <= t_win_hi)])
    print(f"  saved CSV: {out_csv}")

    results[sat] = sat_results

# ── summary table ─────────────────────────────────────────────────────────────
print("\n\n" + "="*55)
print("  SUMMARY — median SNR during signal presence")
print("="*55)
print(f"  {'Satellite':<14} {'Method':<8} {'Median (dB)':>12} {'Peak (dB)':>10}")
print("  " + "-"*46)
for sat, r in results.items():
    for name, v in r.items():
        print(f"  {sat:<14} {name:<8} {v['median']:>12.1f} {v['peak']:>10.1f}")
