"""Analysis.py — DopTrack pipeline benchmarking.

Runs on top of the batch runner's 100-pass sample. For each pass the pipeline
is re-invoked with parameter variants; results are aggregated and written to:

    OUTPUT_ROOT/analysis/figures/snr_*.png
    OUTPUT_ROOT/analysis/figures/interp_*.png
    OUTPUT_ROOT/analysis/figures/bw_*.png
    OUTPUT_ROOT/analysis/report.html          (tabbed interactive report)

Sections
--------
SNR
  - Fixed BW vs estimated BW: scatter of peak/median SNR across passes
  - Smoother comparison (savgol / butterworth / none): box plots of peak, median,
    std of SNR curve per pass

Interpolation
  - gap_threshold sweep: median NaN-fraction vs threshold with percentile band

Bandwidth
  - Distribution histogram of estimated BW across passes
  - Consistency check: BW std across 5 × 80%-frame subsets per pass
  - SNR selection threshold sweep: median estimated BW vs threshold
"""

import os
import html
import random
import datetime as dt
import traceback

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --------------------------------------------------------------------------- #
# Import your pipeline modules
# --------------------------------------------------------------------------- #
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from pipeline.io import read_tuning_freq
from pipeline.process_pass import process_pass
from pipeline.bandwidth import select_strong_frames, estimate_band

# --------------------------------------------------------------------------- #
# Mirror these from Batch_Runner.py
# --------------------------------------------------------------------------- #
IQ_DIR      = r"X:\data\L0"
DAT_DIR     = r"X:\data\L1B"
OUTPUT_ROOT = r"C:\Users\glute\Desktop\Project Python Folder\Data Dopptrack\results"

F_S          = 25_000
N            = 2 ** 12
OVERLAP_FRAC = 0.5
FIXED_BW     = 1200
GAP_THRESHOLD = 7.0
COUNT        = -1

N_SAMPLES    = 100
RANDOM_SEED  = 42

# Sweep parameters
GAP_THRESH_SWEEP   = np.arange(1.0, 21.0, 1.0)       # seconds
BW_SNR_THRESH_SWEEP = np.arange(2.0, 20.0, 2.0)       # dB above noise
N_BW_SUBSETS       = 5
BW_SUBSET_FRAC     = 0.80

SMOOTHERS = ["savgol", "butterworth", None]
SMOOTHER_LABELS = {
    "savgol":      "Savitzky–Golay",
    "butterworth": "Butterworth",
    None:          "None",
}

ANALYSIS_DIR = os.path.join(OUTPUT_ROOT, "analysis")
FIG_DIR      = os.path.join(ANALYSIS_DIR, "figures")


# =========================================================================== #
# Helpers
# =========================================================================== #

def sat_id_from_name(name):
    return name.split("_")[0]


def find_passes(iq_dir, dat_dir, n_samples=100, seed=None):
    import glob
    rng = random.Random(seed)
    candidates = []
    for path_iq in sorted(glob.glob(os.path.join(iq_dir, "*", "*", "*.fc32"))):
        name = os.path.splitext(os.path.basename(path_iq))[0]
        rel  = os.path.relpath(path_iq, iq_dir)
        parts = rel.split(os.sep)
        sat_year = os.path.join(*parts[:-1])
        path_dat = os.path.join(dat_dir, sat_year, name + ".dat")
        path_yml = os.path.join(os.path.dirname(path_iq), name + ".yml")
        if os.path.exists(path_dat) and os.path.exists(path_yml):
            candidates.append((name, path_iq, path_dat, path_yml))
    n = min(n_samples, len(candidates))
    print(f"  [{n}/{len(candidates)} passes selected for analysis]")
    return rng.sample(candidates, n)


def run_variant(path_iq, path_dat, f_tune, **kwargs):
    """Call process_pass with merged kwargs; return result or None on failure."""
    base = dict(
        f_s=F_S, N=N, overlap_frac=OVERLAP_FRAC,
        fixed_bw=FIXED_BW, bw_mode="fixed",
        smoother="savgol", gap_threshold=GAP_THRESHOLD,
        count=COUNT, also_estimate_bw=True,
    )
    base.update(kwargs)
    try:
        return process_pass(path_iq, path_dat, f_tune, **base)
    except Exception:
        return None


def savefig(fig, filename):
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# =========================================================================== #
# SNR analysis
# =========================================================================== #

def analyse_snr(passes, tuning_cache):
    """
    Returns dict with keys:
      bw_mode_rows  : list of dicts per pass comparing fixed vs estimated BW
      smoother_rows : list of dicts per pass comparing smoothers
    Also saves figures.
    """
    print("\n[SNR] running variants ...")
    bw_mode_rows  = []
    smoother_rows = []

    for name, path_iq, path_dat, path_yml in passes:
        sat_id = sat_id_from_name(name)
        f_tune = tuning_cache.get(sat_id)
        if f_tune is None:
            continue

        # --- fixed vs estimated BW ---
        row = {"name": name}
        for mode in ("fixed", "estimate"):
            r = run_variant(path_iq, path_dat, f_tune, bw_mode=mode,
                            smoother="savgol", name=name)
            if r is None:
                row[f"peak_{mode}"] = np.nan
                row[f"med_{mode}"]  = np.nan
            else:
                snr = r["snr_db"]
                row[f"peak_{mode}"] = float(np.nanmax(snr))  if np.any(np.isfinite(snr)) else np.nan
                row[f"med_{mode}"]  = float(np.nanmedian(snr)) if np.any(np.isfinite(snr)) else np.nan
        bw_mode_rows.append(row)

        # --- smoother comparison ---
        srow = {"name": name}
        for sm in SMOOTHERS:
            r = run_variant(path_iq, path_dat, f_tune, bw_mode="fixed",
                            smoother=sm, name=name)
            key = str(sm)
            if r is None:
                srow[f"peak_{key}"]  = np.nan
                srow[f"med_{key}"]   = np.nan
                srow[f"std_{key}"]   = np.nan
            else:
                snr = r["snr_db"]
                fin = snr[np.isfinite(snr)]
                srow[f"peak_{key}"]  = float(np.max(fin))    if len(fin) else np.nan
                srow[f"med_{key}"]   = float(np.median(fin)) if len(fin) else np.nan
                srow[f"std_{key}"]   = float(np.std(fin))    if len(fin) else np.nan
        smoother_rows.append(srow)

    # ---- figures ----
    fig_paths = []

    # scatter: fixed vs estimated BW
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, label in zip(axes,
                                  [("peak_fixed", "peak_estimate"),
                                   ("med_fixed",  "med_estimate")],
                                  ["Peak SNR (dB)", "Median SNR (dB)"]):
        xk, yk = metric
        xs = [r[xk] for r in bw_mode_rows]
        ys = [r[yk] for r in bw_mode_rows]
        ax.scatter(xs, ys, s=18, alpha=0.6, color="#5ec8f2")
        lo = min(np.nanmin(xs), np.nanmin(ys))
        hi = max(np.nanmax(xs), np.nanmax(ys))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y = x")
        ax.set_xlabel(f"Fixed BW  —  {label}")
        ax.set_ylabel(f"Estimated BW  —  {label}")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle("SNR: Fixed BW vs Estimated BW", fontsize=13)
    fig.tight_layout()
    fig_paths.append(savefig(fig, "snr_bw_mode_scatter.png"))

    # box plots: smoother comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [("peak", "Peak SNR (dB)"), ("med", "Median SNR (dB)"), ("std", "SNR std (dB)")]
    colors = ["#5ec8f2", "#f2b45e", "#6fdc8c"]
    for ax, (metric_prefix, ylabel) in zip(axes, metrics):
        data   = [[r[f"{metric_prefix}_{str(sm)}"] for r in smoother_rows
                   if np.isfinite(r[f"{metric_prefix}_{str(sm)}"])]
                  for sm in SMOOTHERS]
        labels = [SMOOTHER_LABELS[sm] for sm in SMOOTHERS]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("SNR: Smoother Comparison (fixed BW)", fontsize=13)
    fig.tight_layout()
    fig_paths.append(savefig(fig, "snr_smoother_boxplots.png"))

    print(f"  [SNR] done — {len(bw_mode_rows)} passes")
    return dict(bw_mode_rows=bw_mode_rows, smoother_rows=smoother_rows,
                fig_paths=fig_paths)


# =========================================================================== #
# Interpolation analysis
# =========================================================================== #

def analyse_interpolation(passes, tuning_cache):
    """Sweep gap_threshold, record NaN fraction per pass per threshold."""
    from viz.tests import check_interpolation_gaps

    print("\n[Interp] sweeping gap_threshold ...")
    # shape: (n_passes, n_thresholds)
    nan_fracs = []
    n_big_gaps_mat = []

    valid_passes = 0
    for name, path_iq, path_dat, path_yml in passes:
        sat_id = sat_id_from_name(name)
        f_tune = tuning_cache.get(sat_id)
        if f_tune is None:
            continue

        # run pipeline once (gap_threshold doesn't affect the raw interp, only
        # the check — so we run once and re-check at each threshold)
        r = run_variant(path_iq, path_dat, f_tune, bw_mode="fixed",
                        smoother="savgol", gap_threshold=GAP_THRESHOLD, name=name)
        if r is None:
            continue

        t_ax    = r["t_ax"]
        f_interp = r["f_interp"]
        t_bf    = r["t_bf"]

        row_frac = []
        row_gaps = []
        for thresh in GAP_THRESH_SWEEP:
            # re-run interpolation check at each threshold
            chk = check_interpolation_gaps(t_ax, f_interp, t_bf, thresh)
            total = chk["n_nan"] + chk["n_valid"]
            frac  = chk["n_nan"] / total if total > 0 else 0.0
            row_frac.append(frac)
            row_gaps.append(chk["n_big_gaps"])
        nan_fracs.append(row_frac)
        n_big_gaps_mat.append(row_gaps)
        valid_passes += 1

    nan_fracs      = np.array(nan_fracs)       # (n_passes, n_thresh)
    n_big_gaps_mat = np.array(n_big_gaps_mat)

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    med  = np.median(nan_fracs, axis=0)
    p10  = np.percentile(nan_fracs, 10, axis=0)
    p90  = np.percentile(nan_fracs, 90, axis=0)
    ax.fill_between(GAP_THRESH_SWEEP, p10, p90, alpha=0.25, color="#5ec8f2", label="10–90th pct")
    ax.plot(GAP_THRESH_SWEEP, med, "-o", ms=4, color="#5ec8f2", label="median")
    ax.set_xlabel("gap_threshold (s)")
    ax.set_ylabel("NaN fraction of frames")
    ax.set_title("NaN coverage vs gap threshold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    med_gaps = np.median(n_big_gaps_mat, axis=0)
    p10_gaps = np.percentile(n_big_gaps_mat, 10, axis=0)
    p90_gaps = np.percentile(n_big_gaps_mat, 90, axis=0)
    ax.fill_between(GAP_THRESH_SWEEP, p10_gaps, p90_gaps, alpha=0.25, color="#f2b45e",
                    label="10–90th pct")
    ax.plot(GAP_THRESH_SWEEP, med_gaps, "-o", ms=4, color="#f2b45e", label="median")
    ax.set_xlabel("gap_threshold (s)")
    ax.set_ylabel("number of large gaps detected")
    ax.set_title("Large-gap count vs gap threshold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle(f"Interpolation: gap_threshold sweep  (n={valid_passes} passes)", fontsize=13)
    fig.tight_layout()
    fig_path = savefig(fig, "interp_gap_sweep.png")

    print(f"  [Interp] done — {valid_passes} passes")
    return dict(nan_fracs=nan_fracs, n_big_gaps_mat=n_big_gaps_mat,
                fig_paths=[fig_path])


# =========================================================================== #
# Bandwidth analysis
# =========================================================================== #

def analyse_bandwidth(passes, tuning_cache):
    print("\n[BW] running analysis ...")
    rng = random.Random(RANDOM_SEED + 1)

    bw_estimates  = []     # one float per pass
    bw_stds       = []     # consistency: std of N_BW_SUBSETS estimates per pass
    thresh_matrix = []     # (n_passes, n_thresh_sweep)

    for name, path_iq, path_dat, path_yml in passes:
        sat_id = sat_id_from_name(name)
        f_tune = tuning_cache.get(sat_id)
        if f_tune is None:
            continue

        r = run_variant(path_iq, path_dat, f_tune, bw_mode="estimate",
                        smoother="savgol", name=name)
        if r is None:
            continue

        bw_est = r.get("bw_estimated_hz", np.nan)
        bw_estimates.append(float(bw_est) if np.isfinite(bw_est) else np.nan)

        pwr      = r["pwr"]          # shape (n_bins, n_frames)
        f_ax     = r["f_ax"]
        f_interp = r["f_interp"]

        # ---- consistency: subsample strong frames ----
        default_strong = select_strong_frames(pwr, f_ax, f_interp, snr_thresh_db=6.0)
        subset_bws = []
        for _ in range(N_BW_SUBSETS):
            n_sub = max(1, int(len(default_strong) * BW_SUBSET_FRAC))
            sub   = rng.sample(list(default_strong), n_sub)
            edges = estimate_band(pwr, f_ax, f_interp, frame_indices=sub)
            if edges is not None:
                subset_bws.append(abs(edges[1] - edges[0]))
        bw_stds.append(float(np.std(subset_bws)) if len(subset_bws) >= 2 else np.nan)

        # ---- SNR threshold sweep ----
        row_thresh = []
        for snr_thr in BW_SNR_THRESH_SWEEP:
            try:
                strong = select_strong_frames(pwr, f_ax, f_interp, snr_thresh_db=float(snr_thr))
                edges  = estimate_band(pwr, f_ax, f_interp, frame_indices=strong)
                row_thresh.append(abs(edges[1] - edges[0]) if edges is not None else np.nan)
            except Exception:
                row_thresh.append(np.nan)
        thresh_matrix.append(row_thresh)

    bw_estimates  = np.array(bw_estimates,  dtype=float)
    bw_stds       = np.array(bw_stds,       dtype=float)
    thresh_matrix = np.array(thresh_matrix, dtype=float)   # (n_passes, n_thresh)

    # ---- figures ----
    fig_paths = []

    # 1. Histogram of estimated BW
    fig, ax = plt.subplots(figsize=(8, 5))
    finite_bw = bw_estimates[np.isfinite(bw_estimates)]
    ax.hist(finite_bw, bins=25, color="#5ec8f2", edgecolor="#1a2029", alpha=0.85)
    ax.axvline(FIXED_BW, color="red", lw=1.5, ls="--", label=f"fixed BW = {FIXED_BW} Hz")
    ax.set_xlabel("Estimated bandwidth (Hz)")
    ax.set_ylabel("Count")
    ax.set_title(f"Bandwidth distribution  (n={len(finite_bw)} passes)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig_paths.append(savefig(fig, "bw_histogram.png"))

    # 2. Consistency: std of subset estimates per pass (sorted)
    fig, ax = plt.subplots(figsize=(10, 4))
    finite_std = bw_stds[np.isfinite(bw_stds)]
    ax.bar(np.arange(len(finite_std)), np.sort(finite_std),
           color="#6fdc8c", edgecolor="#1a2029", alpha=0.85)
    ax.set_xlabel("Pass (sorted by std)")
    ax.set_ylabel("BW estimate std (Hz)")
    ax.set_title(f"BW consistency: std across {N_BW_SUBSETS}×{int(BW_SUBSET_FRAC*100)}% subsets")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig_paths.append(savefig(fig, "bw_consistency.png"))

    # 3. SNR threshold sweep
    fig, ax = plt.subplots(figsize=(9, 5))
    med  = np.nanmedian(thresh_matrix, axis=0)
    p10  = np.nanpercentile(thresh_matrix, 10, axis=0)
    p90  = np.nanpercentile(thresh_matrix, 90, axis=0)
    ax.fill_between(BW_SNR_THRESH_SWEEP, p10, p90, alpha=0.25, color="#f2b45e",
                    label="10–90th pct")
    ax.plot(BW_SNR_THRESH_SWEEP, med, "-o", ms=4, color="#f2b45e", label="median")
    ax.axhline(FIXED_BW, color="red", lw=1, ls="--", label=f"fixed BW = {FIXED_BW} Hz")
    ax.set_xlabel("SNR selection threshold (dB above noise)")
    ax.set_ylabel("Estimated bandwidth (Hz)")
    ax.set_title("BW estimate vs strong-frame SNR threshold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig_paths.append(savefig(fig, "bw_snr_thresh_sweep.png"))

    print(f"  [BW] done — {len(bw_estimates)} passes")
    return dict(bw_estimates=bw_estimates, bw_stds=bw_stds,
                thresh_matrix=thresh_matrix, fig_paths=fig_paths)


# =========================================================================== #
# HTML report
# =========================================================================== #

REPORT_CSS = """
:root {
  --bg:#0f1419; --panel:#1a2029; --line:#2a3340; --ink:#d6dde6;
  --accent:#5ec8f2; --muted:#7d8a99; --green:#6fdc8c; --amber:#f2b45e;
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:14px/1.6 "JetBrains Mono",ui-monospace,monospace; }
header { padding:20px 28px; border-bottom:1px solid var(--line); }
header h1 { margin:0; font-size:20px; letter-spacing:.04em; color:var(--accent); }
header .sub { color:var(--muted); font-size:12px; margin-top:4px; }
nav { display:flex; gap:0; border-bottom:1px solid var(--line); background:var(--panel); }
.tab { padding:12px 22px; cursor:pointer; font-size:13px; color:var(--muted);
       border-bottom:2px solid transparent; user-select:none; }
.tab:hover { color:var(--ink); }
.tab.active { color:var(--accent); border-bottom-color:var(--accent); }
.pane { display:none; padding:24px 28px; }
.pane.active { display:block; }
.fig-row { display:flex; gap:16px; flex-wrap:wrap; margin-bottom:24px; }
.fig-box { flex:1 1 45%; min-width:320px; }
.fig-box h3 { font-size:12px; color:var(--muted); margin:0 0 6px; font-weight:400; }
.fig-box img { width:100%; border:1px solid var(--line); border-radius:4px;
               background:#fff; }
table { border-collapse:collapse; width:100%; font-size:12px; margin-top:12px; }
th,td { padding:6px 10px; text-align:left; border-bottom:1px solid var(--line); white-space:nowrap; }
th { position:sticky; top:0; background:var(--panel); color:var(--accent);
     cursor:pointer; user-select:none; }
th:hover { color:#fff; }
tbody tr:hover { background:#161d27; }
.ok  { color:var(--green); }
.bad { color:#f2776e; }
section-title { display:block; font-size:15px; color:var(--ink);
                margin-bottom:14px; letter-spacing:.03em; }
"""

SORT_JS = """
function sortTable(id, col) {
  const tb = document.querySelector('#'+id+' tbody');
  const rows = Array.from(tb.rows);
  const asc = tb.getAttribute('data-sort') !== col+'-asc';
  rows.sort((a,b) => {
    let x = a.cells[col].innerText, y = b.cells[col].innerText;
    const nx = parseFloat(x), ny = parseFloat(y);
    if (!isNaN(nx) && !isNaN(ny)) { x=nx; y=ny; }
    return (x>y?1:x<y?-1:0)*(asc?1:-1);
  });
  rows.forEach(r => tb.appendChild(r));
  tb.setAttribute('data-sort', col+(asc?'-asc':'-desc'));
}
function showTab(id) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.pane').forEach(p => p.classList.remove('active'));
  document.querySelector('.tab[data-tab="'+id+'"]').classList.add('active');
  document.getElementById(id).classList.add('active');
}
"""


def _img_tag(filename, alt=""):
    return f'<img src="figures/{html.escape(filename)}" alt="{html.escape(alt)}">'


def _table(table_id, cols, rows_data):
    """Render a sortable HTML table."""
    header = "".join(
        f'<th onclick="sortTable(\'{html.escape(table_id)}\',{i})">'
        f'{html.escape(str(c))}</th>'
        for i, c in enumerate(cols)
    )
    body = []
    for r in rows_data:
        cells = "".join(f"<td>{html.escape(str(r.get(c,''))[:40])}</td>" for c in cols)
        body.append(f"<tr>{cells}</tr>")
    return (f'<div style="overflow:auto;max-height:320px">'
            f'<table id="{html.escape(table_id)}"><thead><tr>{header}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def _round_row(r, keys, digits=3):
    out = {"name": r.get("name", "")}
    for k in keys:
        v = r.get(k, "")
        try:
            fv = float(v)
            out[k] = round(fv, digits) if np.isfinite(fv) else "NaN"
        except (TypeError, ValueError):
            out[k] = v
    return out


def write_html_report(snr_data, interp_data, bw_data, n_passes):
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    # ---- SNR pane ----
    bw_cols = ["name", "peak_fixed", "med_fixed", "peak_estimate", "med_estimate"]
    bw_rows = [_round_row(r, bw_cols[1:]) for r in snr_data["bw_mode_rows"]]

    sm_cols = (["name"]
               + [f"peak_{str(sm)}" for sm in SMOOTHERS]
               + [f"med_{str(sm)}"  for sm in SMOOTHERS]
               + [f"std_{str(sm)}"  for sm in SMOOTHERS])
    sm_rows = [_round_row(r, sm_cols[1:]) for r in snr_data["smoother_rows"]]

    snr_pane = f"""
<span class="section-title">Fixed BW vs Estimated BW</span>
<div class="fig-row">
  <div class="fig-box"><h3>Peak & Median SNR scatter</h3>
    {_img_tag("snr_bw_mode_scatter.png", "BW mode scatter")}</div>
</div>
<span class="section-title">Smoother comparison</span>
<div class="fig-row">
  <div class="fig-box"><h3>Box plots</h3>
    {_img_tag("snr_smoother_boxplots.png", "smoother boxplots")}</div>
</div>
<span class="section-title">Per-pass table: BW mode</span>
{_table("t_snr_bw", bw_cols, bw_rows)}
<span class="section-title" style="margin-top:18px">Per-pass table: smoothers</span>
{_table("t_snr_sm", sm_cols, sm_rows)}
"""

    # ---- Interpolation pane ----
    interp_pane = f"""
<span class="section-title">gap_threshold sweep</span>
<div class="fig-row">
  <div class="fig-box"><h3>NaN fraction &amp; large-gap count vs threshold</h3>
    {_img_tag("interp_gap_sweep.png", "gap threshold sweep")}</div>
</div>
"""

    # ---- BW pane ----
    bw_pane = f"""
<span class="section-title">Estimated BW distribution</span>
<div class="fig-row">
  <div class="fig-box"><h3>Histogram across {n_passes} passes</h3>
    {_img_tag("bw_histogram.png", "BW histogram")}</div>
  <div class="fig-box"><h3>Consistency (subset std)</h3>
    {_img_tag("bw_consistency.png", "BW consistency")}</div>
</div>
<span class="section-title">SNR selection threshold sweep</span>
<div class="fig-row">
  <div class="fig-box"><h3>Estimated BW vs SNR threshold</h3>
    {_img_tag("bw_snr_thresh_sweep.png", "BW SNR threshold sweep")}</div>
</div>
"""

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>DopTrack Pipeline Analysis</title>
<style>{REPORT_CSS}</style>
</head><body>
<header>
  <h1>DopTrack — Pipeline Analysis</h1>
  <div class="sub">{n_passes} passes · generated {generated}</div>
</header>
<nav>
  <div class="tab active" data-tab="pane-snr"   onclick="showTab('pane-snr')">SNR</div>
  <div class="tab"        data-tab="pane-interp" onclick="showTab('pane-interp')">Interpolation</div>
  <div class="tab"        data-tab="pane-bw"     onclick="showTab('pane-bw')">Bandwidth</div>
</nav>
<div id="pane-snr"   class="pane active">{snr_pane}</div>
<div id="pane-interp" class="pane">{interp_pane}</div>
<div id="pane-bw"    class="pane">{bw_pane}</div>
<script>{SORT_JS}</script>
</body></html>"""

    path = os.path.join(ANALYSIS_DIR, "report.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    print(f"\n  report -> {path}")
    return path


# =========================================================================== #
# Entry point
# =========================================================================== #

def run():
    os.makedirs(FIG_DIR, exist_ok=True)

    passes = find_passes(IQ_DIR, DAT_DIR, n_samples=N_SAMPLES, seed=RANDOM_SEED)

    # build tuning cache for all selected passes up front
    tuning_cache = {}
    for name, path_iq, path_dat, path_yml in passes:
        sat_id = sat_id_from_name(name)
        if sat_id not in tuning_cache:
            try:
                tuning_cache[sat_id] = read_tuning_freq(path_yml)
            except Exception as e:
                print(f"  [warn] tuning read failed for {sat_id}: {e}")

    snr_data    = analyse_snr(passes, tuning_cache)
    interp_data = analyse_interpolation(passes, tuning_cache)
    bw_data     = analyse_bandwidth(passes, tuning_cache)

    write_html_report(snr_data, interp_data, bw_data, n_passes=len(passes))
    print("\nanalysis complete.")


if __name__ == "__main__":
    run()