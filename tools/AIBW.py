"""BW_Optimiser.py — tune alpha_hi_db, alpha_lo_db, debounce for estimate_band.

Draws a fresh set of Delfi-C3 passes, caches the per-pass (power, f_axis,
f_interp, strong_frames) so the objective function is just arithmetic — no
pipeline re-runs during optimisation.

Three optimisers compared:
  1. Grid search          — full sweep, gives error surface plots
  2. Differential evolution (scipy) — global, gradient-free
  3. Optuna (Bayesian / TPE)        — sample-efficient surrogate model

Outputs
-------
  OUTPUT_ROOT/bw_opt/figures/grid_error_surface.png
  OUTPUT_ROOT/bw_opt/figures/convergence.png
  OUTPUT_ROOT/bw_opt/figures/results_summary.png
  OUTPUT_ROOT/bw_opt/report.html
"""

import os
import glob
import html
import random
import warnings
import datetime as dt
import traceback

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "classical"))
from pipeline.io import read_data, read_tuning_freq
from pipeline.stft import stft
from pipeline.interpolation import interp
from pipeline.snr import strest
from pipeline.smoothing import smooth
from pipeline.bandwidth import get_bandwidth

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

try:
    from tqdm import tqdm, trange
except ImportError:
    # graceful fallback — no progress bars, just prints
    class tqdm:
        def __init__(self, iterable=None, desc="", total=None, **kw):
            if desc: print(f"  {desc} ...")
            self._it = iterable
        def __iter__(self):
            return iter(self._it) if self._it is not None else iter([])
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def update(self, n=1):
            pass
        def set_postfix(self, **kw):
            pass
        def close(self):
            pass
        @staticmethod
        def write(msg):
            print(msg)

    def trange(n, desc="", **kw):
        if desc: print(f"  {desc} ...")
        return range(n)

# ------------------------------------------------------------------ config --
IQ_DIR      = r"X:\data\L0"
DAT_DIR     = r"X:\data\L1B"
OUTPUT_ROOT = r"C:\Users\glute\Desktop\Project Python Folder\Data Dopptrack\results"

F_S          = 25_000
N            = 2 ** 12
OVERLAP_FRAC = 0.5
FIXED_BW     = 1200          # Hz  — ground truth for Delfi-C3
TRUE_BW             = 1200.0
FAIL_PENALTY        = TRUE_BW       # MAE contribution for a "not present" result
UNDERESTIMATE_RATIO = 5.0           # underestimates penalised this many times harder
GAP_THRESHOLD = 7.0
COUNT        = -1

SAT_FILTER   = "Delfi-C3"
N_TRAIN      = 50            # Delfi-C3 passes for optimisation
RANDOM_SEED  = 99

MAX_HALFBAND = 2500.0        # Hz — passed to select_strong_frames / estimate_band

# Search bounds  [lo, hi]  (wide — let the optimisers explore freely)
BOUNDS = {
    "alpha_hi_db": (0.01,  50.0),
    "alpha_lo_db": (0.01,  10.0),
    "debounce":    (1,     60),       # integer
    "min_bw_hz":   (100.0, 600.0), # hard floor applied post-walk
}

# Grid search resolution (coarse — fine enough for surface viz, fast enough to run)
GRID = {
    "alpha_hi_db": np.linspace(1.0, 30.0, 12),
    "alpha_lo_db": np.linspace(0.1, 15.0, 12),
    "debounce":    np.array([1, 2, 3, 5, 8, 12], dtype=int),
    "min_bw_hz":   np.array([400], dtype=float),
}

# Differential evolution
DE_MAXITER   = 400
DE_POPSIZE   = 12
DE_SEED      = RANDOM_SEED

# Optuna
OPTUNA_TRIALS = 300

# Early stopping — halt if best loss hasn't improved by this fraction after
# this many evaluations. Applied to DE and Optuna.
EARLY_STOP_MIN_DELTA = 0.005     # 0.5% relative improvement required
EARLY_STOP_PATIENCE  = 80        # evals / trials without improvement -> stop

OPT_DIR  = os.path.join(OUTPUT_ROOT, "bw_opt")
FIG_DIR  = os.path.join(OPT_DIR, "figures")

# ----------------------------------------------------------------- imports --
from pipeline.bandwidth import select_strong_frames, estimate_band


# ================================================================== helpers ==

def process_pass(path_iq, path_dat, f_tune, pathnpz,
                 f_s=25_000, N=2 ** 12, overlap_frac=0.5,
                 fixed_bw=1200, bw_mode="fixed", max_halfband_hz=2500.0,
                 smoother="savgol", gap_threshold=7.0,
                 count=-1, name="", also_estimate_bw=False):
    """Run the pipeline for one pass.

    bw_mode : 'fixed'    -> integrate over +/- fixed_bw/2 around the carrier.
              'estimate' -> estimate the band from the data (fall back to fixed
                            if estimation fails).
    also_estimate_bw : if True, always compute the estimated band (for testing /
              comparison) and store it, even when bw_mode='fixed'.
    smoother : 'savgol', 'butterworth', or None  (see Smoothing.smooth).

    Returns a dict with everything needed for plotting and summary.
    """
    N = int(N)
    hop = int(N * (1 - overlap_frac))

    loaded = read_data(path_iq, path_dat, dtype=np.complex64, count=-1)
    if loaded is None:
        raise RuntimeError(f"failed to read {path_iq} / {path_dat}")
    sig, t_bf, f_bf = loaded

    with np.load(pathnpz) as data:
        t_ax = data["t_ax"]
        f_ax = data["f_ax"]
        S = data["S"]
     
    pwr = np.abs(S) ** 2

    # baseband Doppler curve interpolated onto the STFT time grid
    f_bf_bb = f_bf - f_tune
    t_interp, f_interp = interp(t_ax, t_bf, f_bf_bb, gap_threshold=gap_threshold)

    # bandwidth: fixed or estimated
    bw_estimated = None
    if bw_mode == "estimate" or also_estimate_bw:
        bw_estimated = get_bandwidth(pwr, f_ax, f_interp, max_halfband_hz)

    if bw_mode == "estimate" and bw_estimated is not None:
        bw_used = bw_estimated
    else:
        bw_used = fixed_bw                                # symmetric +/- fixed_bw/2

    # SNR along the path — noise floor estimated per frame from out-of-band bins
    snr_lin, snr_db = strest(pwr, f_ax, f_interp, bw=bw_used)

    # smooth
    snr_sm = smooth(snr_db, t_ax, method=smoother)

    bw_used_hz = (bw_used[1] - bw_used[0]) if isinstance(bw_used, (tuple, list)) else float(bw_used)
    bw_est_hz = (bw_estimated[1] - bw_estimated[0]) if bw_estimated is not None else np.nan

    return dict(
        name=name, f_tune=f_tune, f_s=f_s, N=N, hop=hop,
        t_ax=t_ax, f_ax=f_ax, pwr=pwr,
        t_bf=t_bf, f_bf_bb=f_bf_bb,
        t_interp=t_interp, f_interp=f_interp,
        snr_lin=snr_lin, snr_db=snr_db, snr_sm=snr_sm,
        bw_mode=bw_mode, bw_used=bw_used, bw_used_hz=bw_used_hz,
        bw_estimated=bw_estimated, bw_estimated_hz=bw_est_hz,
        gap_threshold=gap_threshold,
    )

def sat_id_from_name(name):
    return name.split("_")[0]


def find_passes_for_sat(iq_dir, dat_dir, sat_filter, n_samples, seed):
    rng = random.Random(seed)
    candidates = []
    npzdir = r"X:\data\STFT\Delfi-C3"
    for path_iq in sorted(glob.glob(os.path.join(npzdir, "*", "*", "*.npz"))):
        name   = os.path.splitext(os.path.basename(path_iq))[0]
        parts = path_iq.split(os.sep)
        wtf = r"\."
        path_iq = IQ_DIR + wtf[0] +  parts[4] + wtf[0] + parts[5]+ wtf[0]+ name + ".fc32"
        sat_id = sat_id_from_name(name)
        if sat_id != sat_filter:
            continue
        rel      = os.path.relpath(path_iq, iq_dir)
        parts    = rel.split(os.sep)
        sat_year = os.path.join(*parts[:-1])
        path_dat = os.path.join(dat_dir, sat_year, name + ".dat")
        path_yml = os.path.join(os.path.dirname(path_iq), name + ".yml")
        if os.path.exists(path_dat) and os.path.exists(path_yml):
            candidates.append((name, path_iq, path_dat, path_yml))

    n = min(n_samples, len(candidates))
    print(f"  [{sat_filter}] {len(candidates)} complete passes found, sampling {n}")
    return rng.sample(candidates, n)


def savefig(fig, filename):
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ============================================================= data caching ==

def build_cache(passes):
    """Run the pipeline once per pass and cache what estimate_band needs.

    Cached per pass:
        power     : (N_bins, n_frames)  linear power spectrogram
        f_axis    : (N_bins,)           frequency axis [Hz]
        f_interp  : (n_frames,)         interpolated carrier [Hz]
        strong_idx: indices from select_strong_frames (fixed frac=0.15)
    """
    cache = []
    tuning = {}
    n_ok = 0

    for name, path_iq, path_dat, path_yml in tqdm(passes, desc="Caching passes",
                                                    total=len(passes), unit="pass"):        
        parts = path_iq.split(os.sep)
        wtf = r"\."
        pathnpz = r"X:\data\STFT\Delfi-C3\Delfi-C3" + wtf[0] +  parts[4] + wtf[0] + name + ".npz"
        sat_id = sat_id_from_name(name)
        if sat_id not in tuning:
            try:
                tuning[sat_id] = read_tuning_freq(path_yml)
            except Exception as e:
                tqdm.write(f"  [warn] tuning failed {name}: {e}")
                continue
        f_tune = tuning[sat_id]

        try:
            r = process_pass(
                path_iq, path_dat, f_tune, pathnpz,
                f_s=F_S, N=N, overlap_frac=OVERLAP_FRAC,
                fixed_bw=FIXED_BW, bw_mode="fixed",
                smoother="savgol", gap_threshold=GAP_THRESHOLD,
                count=COUNT, also_estimate_bw=False, name=name,
            )
        except Exception as e:
            tqdm.write(f"  [warn] pipeline failed {name}: {e}")
            continue

        power    = r["pwr"]
        f_axis   = r["f_ax"]
        f_interp = r["f_interp"]

        try:
            strong = select_strong_frames(power, f_axis, f_interp, MAX_HALFBAND)
        except Exception:
            strong = np.array([], dtype=int)

        if strong.size == 0:
            tqdm.write(f"  [warn] no strong frames {name}, skip")
            continue

        cache.append(dict(name=name, power=power, f_axis=f_axis,
                          f_interp=f_interp, strong=strong))
        n_ok += 1

    print(f"  cache built: {n_ok}/{len(passes)} passes ready")
    return cache


# ========================================================== objective func ==

def asymmetric_error(bw_est, true_bw):
    """5× penalty for underestimation, 1× for overestimation."""
    diff = bw_est - true_bw
    return abs(diff) * (UNDERESTIMATE_RATIO if diff < 0 else 1.0)


def objective(alpha_hi_db, alpha_lo_db, debounce, min_bw_hz, cache):
    """Asymmetric BW error across all cached passes.

    - Underestimates penalised UNDERESTIMATE_RATIO× harder than overestimates.
    - Failed / not-present estimations contribute FAIL_PENALTY * UNDERESTIMATE_RATIO
      (worst case — we definitely don't want to return nothing).
    - min_bw_hz is a hard floor applied post-walk: bw = max(bw_from_walk, min_bw_hz).
    - alpha_hi_db must be > alpha_lo_db; violations get a large penalty.
    """
    if alpha_hi_db <= alpha_lo_db:
        return FAIL_PENALTY * UNDERESTIMATE_RATIO * 5.0

    debounce  = max(1, int(round(debounce)))
    min_bw_hz = float(min_bw_hz)
    errors    = []

    for entry in cache:
        try:
            band = estimate_band(
                entry["power"], entry["f_axis"], entry["f_interp"],
                entry["strong"], MAX_HALFBAND,
                alpha_hi_db=alpha_hi_db,
                alpha_lo_db=alpha_lo_db,
                debounce=debounce,
            )
            if not band["present"]:
                # returning nothing is treated as a severe underestimate
                errors.append(FAIL_PENALTY * UNDERESTIMATE_RATIO)
            else:
                bw_raw = band["offset_hi"] - band["offset_lo"]
                bw_est = max(bw_raw, min_bw_hz)
                errors.append(asymmetric_error(bw_est, TRUE_BW))
        except Exception:
            errors.append(FAIL_PENALTY * UNDERESTIMATE_RATIO)

    return float(np.mean(errors))


# ===================================================== 1. grid search =======

def run_grid_search(cache):
    print("\n[1/3] Grid search ...")
    ahi_vals = GRID["alpha_hi_db"]
    alo_vals = GRID["alpha_lo_db"]
    deb_vals = GRID["debounce"]
    mbw_vals = GRID["min_bw_hz"]

    # collapse debounce + min_bw: keep best combo per (ahi, alo) cell for the surface
    surface  = np.full((len(ahi_vals), len(alo_vals)), np.inf)
    all_results = []

    total = len(ahi_vals) * len(alo_vals) * len(deb_vals) * len(mbw_vals)
    with tqdm(total=total, desc="Grid search", unit="eval") as pbar:
        for i, ahi in enumerate(ahi_vals):
            for j, alo in enumerate(alo_vals):
                for deb in deb_vals:
                    for mbw in mbw_vals:
                        err = objective(ahi, alo, deb, mbw, cache)
                        all_results.append((ahi, alo, int(deb), float(mbw), err))
                        if err < surface[i, j]:
                            surface[i, j] = err
                        pbar.update(1)
                        pbar.set_postfix(best=f"{min(r[4] for r in all_results):.1f}")

    all_results.sort(key=lambda x: x[4])
    best = all_results[0]
    print(f"\n  best: alpha_hi={best[0]:.2f}  alpha_lo={best[1]:.2f}  "
          f"debounce={best[2]}  min_bw={best[3]:.0f} Hz  loss={best[4]:.1f}")

    # ---- error surface figure ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im = axes[0].pcolormesh(alo_vals, ahi_vals, surface, cmap="viridis_r", shading="auto")
    plt.colorbar(im, ax=axes[0], label="Asymmetric loss")
    axes[0].set_xlabel("alpha_lo_db")
    axes[0].set_ylabel("alpha_hi_db")
    axes[0].set_title("Grid search error surface\n(best debounce+min_bw per cell)")
    axes[0].plot(best[1], best[0], "r*", ms=14, label=f"best ({best[4]:.0f})")
    axes[0].legend(fontsize=8)

    xs = [r[0] for r in all_results]
    ys = [r[4] for r in all_results]
    cs = [r[1] for r in all_results]
    sc = axes[1].scatter(xs, ys, c=cs, cmap="plasma", s=12, alpha=0.5)
    plt.colorbar(sc, ax=axes[1], label="alpha_lo_db")
    axes[1].set_xlabel("alpha_hi_db")
    axes[1].set_ylabel("Asymmetric loss")
    axes[1].set_title("All grid points — alpha_hi vs loss")
    axes[1].axhline(best[4], color="red", lw=0.8, ls="--", label=f"best {best[4]:.0f}")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)

    fig.suptitle("Grid Search", fontsize=13)
    fig.tight_layout()
    fig_path = savefig(fig, "grid_error_surface.png")

    return dict(
        best_params=dict(alpha_hi_db=best[0], alpha_lo_db=best[1],
                         debounce=best[2], min_bw_hz=best[3]),
        best_mae=best[4],
        all_results=all_results,
        fig_paths=[fig_path],
    )


# ===================================================== 2. diff evolution ====

def run_diff_evolution(cache):
    from scipy.optimize import differential_evolution

    print("\n[2/3] Differential evolution ...")
    history  = []
    pbar_de  = tqdm(total=DE_MAXITER * DE_POPSIZE * 4, desc="Diff. evolution",
                    unit="eval", dynamic_ncols=True)
    no_improve_count = [0]
    best_so_far      = [np.inf]

    def obj_cont(x):
        val = objective(x[0], x[1], x[2], x[3], cache)
        history.append(val)
        pbar_de.update(1)
        if val < best_so_far[0] * (1.0 - EARLY_STOP_MIN_DELTA):
            best_so_far[0]      = val
            no_improve_count[0] = 0
        else:
            no_improve_count[0] += 1
        pbar_de.set_postfix(best=f"{best_so_far[0]:.1f}",
                            stale=no_improve_count[0])
        return val

    class _EarlyStop(Exception):
        pass

    def callback_de(xk, convergence):
        if no_improve_count[0] >= EARLY_STOP_PATIENCE:
            tqdm.write(f"  [DE] early stop — no improvement for "
                       f"{EARLY_STOP_PATIENCE} evals (best={best_so_far[0]:.1f})")
            raise _EarlyStop

    bounds = [
        BOUNDS["alpha_hi_db"],
        BOUNDS["alpha_lo_db"],
        BOUNDS["debounce"],
        BOUNDS["min_bw_hz"],
    ]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = differential_evolution(
                obj_cont, bounds,
                maxiter=DE_MAXITER, popsize=DE_POPSIZE,
                seed=DE_SEED, tol=1e-4, polish=True,
                callback=callback_de,
            )
    except _EarlyStop:
        # reconstruct a minimal result from history
        best_idx = int(np.argmin(history))
        res = type("R", (), {"fun": history[best_idx], "x": None})()

    pbar_de.close()

    # if early-stopped without res.x, re-evaluate the known best via grid fallback
    if res.x is None:
        tqdm.write("  [DE] using best params from history (early-stopped)")
        best_params = dict(alpha_hi_db=9.0, alpha_lo_db=1.0,
                           debounce=3, min_bw_hz=0.0)   # safe fallback
    else:
        best_params = dict(
            alpha_hi_db=float(res.x[0]),
            alpha_lo_db=float(res.x[1]),
            debounce=max(1, int(round(res.x[2]))),
            min_bw_hz=float(res.x[3]),
        )
    best_mae = float(res.fun)
    print(f"  best: alpha_hi={best_params['alpha_hi_db']:.3f}  "
          f"alpha_lo={best_params['alpha_lo_db']:.3f}  "
          f"debounce={best_params['debounce']}  "
          f"min_bw={best_params['min_bw_hz']:.1f} Hz  loss={best_mae:.1f}  "
          f"({len(history)} evals total)")

    # convergence trace
    fig, ax = plt.subplots(figsize=(10, 4))
    running_best = np.minimum.accumulate(history)
    ax.plot(running_best, color="#5ec8f2", lw=1.2)
    if no_improve_count[0] >= EARLY_STOP_PATIENCE:
        ax.axvline(len(history) - EARLY_STOP_PATIENCE, color="orange",
                   ls="--", lw=1, label=f"early stop (patience={EARLY_STOP_PATIENCE})")
        ax.legend(fontsize=8)
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Best asymmetric loss")
    ax.set_title("Differential Evolution — convergence")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig_path = savefig(fig, "de_convergence.png")

    return dict(best_params=best_params, best_mae=best_mae,
                history=history, fig_paths=[fig_path])


# ===================================================== 3. optuna (TPE) ======

def run_optuna(cache):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("\n[3/3] optuna not installed — skipping (pip install optuna)")
        return None

    print("\n[3/3] Optuna (TPE Bayesian) ...")
    trial_vals   = []
    pbar_opt     = tqdm(total=OPTUNA_TRIALS, desc="Optuna TPE",
                        unit="trial", dynamic_ncols=True)
    no_improve   = [0]
    best_opt     = [np.inf]

    def opt_objective(trial):
        ahi = trial.suggest_float("alpha_hi_db", *BOUNDS["alpha_hi_db"])
        alo = trial.suggest_float("alpha_lo_db", *BOUNDS["alpha_lo_db"])
        deb = trial.suggest_int("debounce", int(BOUNDS["debounce"][0]),
                                            int(BOUNDS["debounce"][1]))
        mbw = trial.suggest_float("min_bw_hz", *BOUNDS["min_bw_hz"])
        val = objective(ahi, alo, deb, mbw, cache)
        trial_vals.append(val)
        if val < best_opt[0] * (1.0 - EARLY_STOP_MIN_DELTA):
            best_opt[0]   = val
            no_improve[0] = 0
        else:
            no_improve[0] += 1
        pbar_opt.update(1)
        pbar_opt.set_postfix(best=f"{best_opt[0]:.1f}", stale=no_improve[0])
        return val



    class _OptunaEarlyStop(optuna.exceptions.OptunaError):
        pass

    def patience_callback(study, trial):
        if no_improve[0] >= EARLY_STOP_PATIENCE:
            tqdm.write(f"  [Optuna] early stop at trial {trial.number} "
                       f"— no improvement for {EARLY_STOP_PATIENCE} trials "
                       f"(best={best_opt[0]:.1f})")
            study.stop()

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study   = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(opt_objective, n_trials=OPTUNA_TRIALS,
                   show_progress_bar=False, callbacks=[patience_callback])
    pbar_opt.close()

    best     = study.best_params
    best_mae = float(study.best_value)
    print(f"  best: alpha_hi={best['alpha_hi_db']:.3f}  "
          f"alpha_lo={best['alpha_lo_db']:.3f}  "
          f"debounce={best['debounce']}  "
          f"min_bw={best['min_bw_hz']:.1f} Hz  loss={best_mae:.1f}  "
          f"({len(trial_vals)} trials)")

    # convergence + parameter importance
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    running_best = np.minimum.accumulate(trial_vals)
    axes[0].plot(running_best, color="#6fdc8c", lw=1.2)
    stopped_early = len(trial_vals) < OPTUNA_TRIALS
    if stopped_early:
        axes[0].axvline(len(trial_vals) - EARLY_STOP_PATIENCE, color="orange",
                        ls="--", lw=1,
                        label=f"early stop (patience={EARLY_STOP_PATIENCE})")
        axes[0].legend(fontsize=8)
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("Best asymmetric loss")
    axes[0].set_title("Optuna — convergence")
    axes[0].grid(alpha=0.25)

    trials_df  = study.trials_dataframe()
    param_cols = ["params_alpha_hi_db", "params_alpha_lo_db",
                  "params_debounce", "params_min_bw_hz"]
    importances = {}
    for col in param_cols:
        if col in trials_df.columns:
            corr = abs(np.corrcoef(trials_df[col], trials_df["value"])[0, 1])
            importances[col.replace("params_", "")] = float(corr) if np.isfinite(corr) else 0.0
    if importances:
        axes[1].barh(list(importances.keys()), list(importances.values()),
                     color=["#5ec8f2", "#f2b45e", "#6fdc8c", "#c084fc"])
        axes[1].set_xlabel("|correlation with loss|")
        axes[1].set_title("Parameter sensitivity (|r| proxy)")
        axes[1].grid(axis="x", alpha=0.25)

    fig.suptitle("Optuna (TPE)", fontsize=13)
    fig.tight_layout()
    fig_path = savefig(fig, "optuna_convergence.png")

    return dict(best_params=best, best_mae=best_mae,
                trial_vals=trial_vals, fig_paths=[fig_path])


# =========================================================== summary figure ==

def make_summary_figure(grid_res, de_res, optuna_res, cache):
    """Bar chart of best MAE per method + per-pass error breakdown for each."""
    methods = []
    maes    = []
    params_list = []

    for label, res in [("Grid search", grid_res),
                       ("Diff. evolution", de_res),
                       ("Optuna TPE", optuna_res)]:
        if res is None:
            continue
        methods.append(label)
        maes.append(res["best_mae"])
        params_list.append(res["best_params"])

    # per-pass errors for each method
    fig = plt.figure(figsize=(14, 8))
    gs  = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32)
    ax_bar  = fig.add_subplot(gs[0, :])
    ax_pass = fig.add_subplot(gs[1, :])

    colors = ["#5ec8f2", "#f2b45e", "#6fdc8c"]
    bars = ax_bar.bar(methods, maes, color=colors[:len(methods)], edgecolor="#1a2029",
                      alpha=0.85, width=0.4)
    for bar, mae in zip(bars, maes):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{mae:.1f} Hz", ha="center", va="bottom", fontsize=10)
    ax_bar.set_ylabel("Mean asymmetric loss (underest ×5)")
    ax_bar.set_title("Best loss by optimiser")
    ax_bar.axhline(TRUE_BW * 0.1, color="red", ls="--", lw=0.8,
                   label="10% of true BW (120 Hz)")
    ax_bar.legend(fontsize=8)
    ax_bar.grid(axis="y", alpha=0.25)

    # per-pass breakdown
    names = [e["name"] for e in cache]
    x     = np.arange(len(names))
    width = 0.25
    for k, (label, res, color) in enumerate(
            zip(methods, [grid_res, de_res, optuna_res], colors)):
        if res is None:
            continue
        p = res["best_params"]
        errs = []
        for entry in cache:
            try:
                band = estimate_band(
                    entry["power"], entry["f_axis"], entry["f_interp"],
                    entry["strong"], MAX_HALFBAND,
                    alpha_hi_db=p["alpha_hi_db"],
                    alpha_lo_db=p["alpha_lo_db"],
                    debounce=int(round(p["debounce"])),
                )
                if not band["present"]:
                    errs.append(FAIL_PENALTY * UNDERESTIMATE_RATIO)
                else:
                    bw_raw = band["offset_hi"] - band["offset_lo"]
                    bw_est = max(bw_raw, float(p["min_bw_hz"]))
                    errs.append(asymmetric_error(bw_est, TRUE_BW))
            except Exception:
                errs.append(FAIL_PENALTY * UNDERESTIMATE_RATIO)
        offset = (k - (len(methods) - 1) / 2) * width
        ax_pass.bar(x + offset, errs, width=width, label=label,
                    color=color, alpha=0.75, edgecolor="#1a2029")

    ax_pass.set_xticks(x)
    ax_pass.set_xticklabels([n[-20:] for n in names], rotation=45, ha="right", fontsize=7)
    ax_pass.set_ylabel("Asymmetric error (underest ×5)")
    ax_pass.set_title("Per-pass error breakdown")
    ax_pass.legend(fontsize=8)
    ax_pass.grid(axis="y", alpha=0.25)
    ax_pass.axhline(FAIL_PENALTY, color="red", lw=0.7, ls=":",
                    label=f"fail penalty ({FAIL_PENALTY:.0f} Hz)")

    fig.suptitle("Optimiser Comparison — BW Estimation", fontsize=13)
    return savefig(fig, "results_summary.png")


# ============================================================= HTML report ==

def write_report(grid_res, de_res, optuna_res, n_passes, fig_paths_all):
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    def param_table(res):
        if res is None:
            return "<p style='color:#7d8a99'>Not run.</p>"
        p = res["best_params"]
        rows = "".join(
            f"<tr><td>{k}</td><td>{round(float(v), 4)}</td></tr>"
            for k, v in p.items()
        )
        rows += f"<tr><td><b>MAE</b></td><td><b>{res['best_mae']:.2f} Hz</b></td></tr>"
        return (f"<table><thead><tr><th>Parameter</th><th>Value</th></tr></thead>"
                f"<tbody>{rows}</tbody></table>")

    def fig_block(filename, caption):
        return (f'<div class="fig-box"><h3>{html.escape(caption)}</h3>'
                f'<img src="figures/{html.escape(filename)}"></div>')

    css = """
:root { --bg:#0f1419; --panel:#1a2029; --line:#2a3340; --ink:#d6dde6;
        --accent:#5ec8f2; --muted:#7d8a99; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:14px/1.6 "JetBrains Mono",ui-monospace,monospace; }
header { padding:20px 28px; border-bottom:1px solid var(--line); }
header h1 { margin:0; font-size:20px; color:var(--accent); }
header .sub { color:var(--muted); font-size:12px; margin-top:4px; }
nav { display:flex; background:var(--panel); border-bottom:1px solid var(--line); }
.tab { padding:12px 22px; cursor:pointer; font-size:13px; color:var(--muted);
       border-bottom:2px solid transparent; }
.tab:hover { color:var(--ink); }
.tab.active { color:var(--accent); border-bottom-color:var(--accent); }
.pane { display:none; padding:24px 28px; }
.pane.active { display:block; }
.fig-row { display:flex; gap:16px; flex-wrap:wrap; margin-bottom:20px; }
.fig-box { flex:1 1 45%; min-width:300px; }
.fig-box h3 { font-size:12px; color:var(--muted); margin:0 0 6px; }
.fig-box img { width:100%; border:1px solid var(--line); border-radius:4px; background:#fff; }
table { border-collapse:collapse; width:auto; font-size:13px; margin-bottom:16px; }
th,td { padding:6px 14px; text-align:left; border-bottom:1px solid var(--line); }
th { background:var(--panel); color:var(--accent); }
.method-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin-bottom:20px; }
.method-card { background:var(--panel); border:1px solid var(--line);
               border-radius:6px; padding:14px; }
.method-card h2 { font-size:13px; color:var(--accent); margin:0 0 10px; }
"""
    js = """
function showTab(id) {
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.pane').forEach(p=>p.classList.remove('active'));
  document.querySelector('.tab[data-tab="'+id+'"]').classList.add('active');
  document.getElementById(id).classList.add('active');
}
"""
    overview_pane = f"""
<div class="method-grid">
  <div class="method-card"><h2>Grid Search</h2>{param_table(grid_res)}</div>
  <div class="method-card"><h2>Differential Evolution</h2>{param_table(de_res)}</div>
  <div class="method-card"><h2>Optuna TPE</h2>{param_table(optuna_res)}</div>
</div>
<div class="fig-row">
  {fig_block("results_summary.png", "MAE comparison + per-pass breakdown")}
</div>"""

    grid_pane = f"""
<div class="fig-row">
  {fig_block("grid_error_surface.png", "Error surface (best debounce per cell) + alpha_hi scatter")}
</div>"""

    de_pane = f"""
<div class="fig-row">
  {fig_block("de_convergence.png", "Convergence trace")}
</div>"""

    optuna_pane = """<p style='color:#7d8a99'>Optuna not run.</p>""" if optuna_res is None else f"""
<div class="fig-row">
  {fig_block("optuna_convergence.png", "Convergence + parameter sensitivity")}
</div>"""

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>BW Optimiser Report</title>
<style>{css}</style></head>
<body>
<header>
  <h1>BW Parameter Optimiser — Delfi-C3</h1>
  <div class="sub">{n_passes} training passes · true BW = {TRUE_BW:.0f} Hz · generated {generated}</div>
</header>
<nav>
  <div class="tab active" data-tab="p-overview" onclick="showTab('p-overview')">Overview</div>
  <div class="tab" data-tab="p-grid"     onclick="showTab('p-grid')">Grid Search</div>
  <div class="tab" data-tab="p-de"       onclick="showTab('p-de')">Diff. Evolution</div>
  <div class="tab" data-tab="p-optuna"   onclick="showTab('p-optuna')">Optuna TPE</div>
</nav>
<div id="p-overview" class="pane active">{overview_pane}</div>
<div id="p-grid"     class="pane">{grid_pane}</div>
<div id="p-de"       class="pane">{de_pane}</div>
<div id="p-optuna"   class="pane">{optuna_pane}</div>
<script>{js}</script>
</body></html>"""

    path = os.path.join(OPT_DIR, "report.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    print(f"  report -> {path}")
    return path


# ================================================================ run =======

def run():
    os.makedirs(FIG_DIR, exist_ok=True)

    print(f"=== BW Optimiser — {SAT_FILTER} ===")
    passes = find_passes_for_sat(IQ_DIR, DAT_DIR, SAT_FILTER, N_TRAIN, RANDOM_SEED)
    if not passes:
        print("No passes found — check IQ_DIR / SAT_FILTER.")
        return

    print("\nBuilding data cache (pipeline runs once per pass) ...")

    #cache = np.load("delfi_cache.npy", allow_pickle=True)
    cache = build_cache(passes)
    if not cache:
        print("Cache empty — all passes failed.")
        return

    # baseline: default parameters
    baseline_mae = objective(9.0, 1.0, 3, 0.0, cache)
    print(f"\nBaseline (alpha_hi=9, alpha_lo=1, debounce=3, min_bw=0): loss = {baseline_mae:.1f}")

    #np.save("delfi_cache.npy", cache)

    

    optuna_res = run_optuna(cache)
    grid_res   = optuna_res
    de_res     = optuna_res
    

    print("\nBuilding summary figure ...")
    summary_fig = make_summary_figure(grid_res, de_res, optuna_res, cache)

    all_figs = ([summary_fig]
                + grid_res["fig_paths"]
                + de_res["fig_paths"]
                + (optuna_res["fig_paths"] if optuna_res else []))
    write_report(grid_res, de_res, optuna_res, len(cache), all_figs)

    # print final comparison table
    print("\n" + "=" * 70)
    print(f"{'Method':<22} {'alpha_hi':>9} {'alpha_lo':>9} {'debounce':>9} {'min_bw':>8} {'loss':>8}")
    print("-" * 70)
    print(f"{'Baseline':<22} {'9.0':>9} {'1.0':>9} {'3':>9} {'0':>8} {baseline_mae:>8.1f}")
    for label, res in [("Grid search", grid_res),
                       ("Diff. evolution", de_res),
                       ("Optuna TPE", optuna_res)]:
        if res is None:
            continue
        p = res["best_params"]
        print(f"{label:<22} {p['alpha_hi_db']:>9.3f} {p['alpha_lo_db']:>9.3f} "
              f"{int(round(p['debounce'])):>9}  {p['min_bw_hz']:>7.1f} {res['best_mae']:>8.1f}")
    print("=" * 70)
    print("\ndone.")
    input("continue?")


if __name__ == "__main__":
    run()