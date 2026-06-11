"""Batch runner.

Iterates over every .fc32 recording in IQ_DIR, matches the .dat.txt (in DAT_DIR)
and .yml (in IQ_DIR) by basename, runs the pipeline, and writes per-satellite
output:

    OUTPUT_ROOT/<Satellite>/figures/<pass>_pass.png
    OUTPUT_ROOT/<Satellite>/figures/<pass>_test.png   (if RUN_TESTS)
    OUTPUT_ROOT/<Satellite>/summary.csv
    OUTPUT_ROOT/<Satellite>/summary.html             (interactive comparison)
    OUTPUT_ROOT/failures.log

Satellite id = filename up to the first '_'. Tuning frequency is read from the
.yml once per satellite and cached. Failures are logged and the loop continues.
"""

import os
import glob
import csv
import html
import traceback
import datetime as dt

import numpy as np
import matplotlib
matplotlib.use("Agg")        # headless: save figures without a display

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from pipeline.io import read_tuning_freq
from pipeline.process_pass import process_pass
from viz.figures import make_pass_figure
from viz.tests import check_interpolation_gaps, check_bandwidth, make_test_figure


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
IQ_DIR = r"X:\data\L0"
DAT_DIR = r"X:\data\L1B"
OUTPUT_ROOT = r"C:\Users\glute\Desktop\Project Python Folder\Data Dopptrack\results\bwtest"

# pipeline params
F_S = 25_000
N = 2 ** 12
OVERLAP_FRAC = 0.5
FIXED_BW = 1200            # Hz, used when BW_MODE='fixed'
BW_MODE = "fixed"          # 'fixed' or 'estimate'
SMOOTHER = "butterworth"        # 'savgol', 'butterworth', or None
GAP_THRESHOLD = 7.0
COUNT = -1                 # IQ samples to read (-1 = all)

RUN_TESTS = True
# known true bandwidths per satellite id, for the bandwidth test (Hz)
KNOWN_BW = {
    "Delfi-C3": 1200,
    # "FUNcube-1": ...,    # fill in when you test it
}
BW_REL_TOL = 0.20          # 20% tolerance for the bandwidth check


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sat_id_from_name(name):
    """Satellite id = filename up to the first underscore."""
    return name.split("_")[0]


import random

def find_passes(iq_dir, dat_dir, n_samples='all', seed=None):
    """Walk L0/<satellite>/<year>/*.fc32, match against L1B/<satellite>/<year>/*.dat.txt
    and L0/<satellite>/<year>/*.yml, then return a random sample of up to n_samples.
    """
    rng = random.Random(seed)
    candidates = []

    for path_iq in sorted(glob.glob(os.path.join(iq_dir, "*", "*", "*.fc32"))):
        name = os.path.splitext(os.path.basename(path_iq))[0]

        # mirror the satellite/year subfolder structure into dat_dir
        rel = os.path.relpath(path_iq, iq_dir)          # e.g. Delfi-C3\2022\name.fc32
        parts = rel.split(os.sep)                        # ['Delfi-C3', '2022', 'name.fc32']
        sat_year = os.path.join(*parts[:-1])             # 'Delfi-C3\2022'

        path_dat = os.path.join(dat_dir, sat_year, name + ".dat")
        path_yml = os.path.join(os.path.dirname(path_iq), name + ".yml")
        

        if os.path.exists(path_dat) and os.path.exists(path_yml): #and (parts[0]=='Delfi-C3' or parts[0]=='Delfi-n3Xt'):
            print(name)
            candidates.append((name, path_iq, path_dat, path_yml))
        else:
            if not os.path.exists(path_dat):
                print(f"  [skip] missing dat: {path_dat}")
            if not os.path.exists(path_yml):
                print(f"  [skip] missing yml: {path_yml}")

    
    if (n_samples == 'all'):
        n = len(candidates)
    else:
        n = min(n_samples, len(candidates))
    print(f"Found {len(candidates)} complete passes, sampling {n}.")
    return rng.sample(candidates, n)


def summary_row(name, sat_id, result, test_results):
    """Flatten one pass into a dict of summary columns."""
    snr = result["snr_db"]
    finite = np.isfinite(snr)
    row = dict(
        name=name,
        satellite=sat_id,
        duration_s=round(float(result["t_ax"].max() - result["t_ax"].min()), 1),
        n_frames=int(result["t_ax"].size),
        peak_snr_db=round(float(np.nanmax(snr)), 2) if finite.any() else "",
        median_snr_db=round(float(np.nanmedian(snr)), 2) if finite.any() else "",
        bw_mode=result["bw_mode"],
        bw_used_hz=round(float(result["bw_used_hz"]), 1),
        bw_estimated_hz=(round(float(result["bw_estimated_hz"]), 1)
                         if np.isfinite(result["bw_estimated_hz"]) else ""),
    )
    if test_results:
        ig = test_results.get("interp", {})
        bw = test_results.get("bw", {})
        row.update(
            interp_ok=ig.get("ok", ""),
            interp_n_gaps=ig.get("n_big_gaps", ""),
            bw_test_ok=bw.get("ok", ""),
            bw_rel_error=(round(bw.get("rel_error"), 3)
                          if bw.get("rel_error") is not None
                          and np.isfinite(bw.get("rel_error", np.nan)) else ""),
        )
    return row


# --------------------------------------------------------------------------- #
# output writers
# --------------------------------------------------------------------------- #
def write_csv(path, rows):
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_html(path, sat_id, rows):
    """Self-contained interactive summary: a sortable table where clicking a row
    shows that pass's figures (pass figure + test figure if present). Figures are
    linked relative to this file (figures/<name>_pass.png).
    """
    if not rows:
        return
    cols = list(rows[0].keys())

    # build table rows; data-name drives the figure panel
    body = []
    for r in rows:
        cells = "".join(f"<td>{html.escape(str(r[c]))}</td>" for c in cols)
        if sat_id == "all":
            body.append(f'<tr data-name="{html.escape(r["name"])}" data-sat="{html.escape(r["satellite"])}">{cells}</tr>')
        else:
            body.append(f'<tr data-name="{html.escape(r["name"])}">{cells}</tr>')
    header = "".join(f"<th onclick=\"sortTable({i})\">{html.escape(c)}</th>"
                     for i, c in enumerate(cols))

    title = sat_id if sat_id != "all" else "All Satellites"
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{html.escape(title)} - pass summary</title>
<style>
  :root {{ --bg:#0f1419; --panel:#1a2029; --line:#2a3340; --ink:#d6dde6;
           --accent:#5ec8f2; --muted:#7d8a99; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
          font:14px/1.5 "JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace; }}
  header {{ padding:18px 24px; border-bottom:1px solid var(--line); }}
  h1 {{ margin:0; font-size:18px; letter-spacing:.04em; }}
  .sub {{ color:var(--muted); font-size:12px; margin-top:4px; }}
  .wrap {{ display:flex; gap:0; height:calc(100vh - 64px); }}
  .left {{ flex:1 1 60%; overflow:auto; border-right:1px solid var(--line); }}
  .right {{ flex:1 1 40%; overflow:auto; padding:16px; }}
  table {{ border-collapse:collapse; width:100%; font-size:12.5px; }}
  th,td {{ padding:7px 10px; text-align:left; border-bottom:1px solid var(--line);
           white-space:nowrap; }}
  th {{ position:sticky; top:0; background:var(--panel); cursor:pointer;
        user-select:none; color:var(--accent); }}
  th:hover {{ color:#fff; }}
  tbody tr {{ cursor:pointer; }}
  tbody tr:hover {{ background:#161d27; }}
  tbody tr.sel {{ background:#143041; outline:1px solid var(--accent); }}
  .fig {{ margin-bottom:18px; }}
  .fig h3 {{ font-size:12px; color:var(--muted); margin:0 0 6px; font-weight:400; }}
  .fig img {{ width:100%; border:1px solid var(--line); border-radius:4px;
              background:#fff; }}
  .hint {{ color:var(--muted); padding:24px; }}
  .ok {{ color:#6fdc8c; }} .bad {{ color:#f2776e; }}
</style></head>
<body>
<header>
  <h1>{html.escape(title)} &mdash; pass summary</h1>
  <div class="sub">{len(rows)} passes &middot; generated {generated} &middot;
    click a column header to sort, a row to view figures</div>
</header>
<div class="wrap">
  <div class="left">
    <table id="t"><thead><tr>{header}</tr></thead>
    <tbody>{''.join(body)}</tbody></table>
  </div>
  <div class="right" id="panel"><div class="hint">Select a pass to view its figures.</div></div>
</div>
<script>
  const rows = document.querySelectorAll('#t tbody tr');
  rows.forEach(r => r.addEventListener('click', () => {{
    rows.forEach(x => x.classList.remove('sel'));
    r.classList.add('sel');
    const n = r.getAttribute('data-name');
    const s = r.getAttribute('data-sat');
    const prefix = s ? s + '/figures/' : 'figures/';
    document.getElementById('panel').innerHTML =
      '<div class="fig"><h3>spectrogram + SNR</h3>'
      + '<img src="' + prefix + n + '_pass.png" '
      + 'onerror="this.parentNode.innerHTML=\\'<h3 class=bad>pass figure missing</h3>\\'"></div>'
      + '<div class="fig"><h3>tests</h3>'
      + '<img src="' + prefix + n + '_test.png" '
      + 'onerror="this.parentNode.style.display=\\'none\\'"></div>';
  }}));
  function sortTable(col) {{
    const tb = document.querySelector('#t tbody');
    const arr = Array.from(tb.rows);
    const asc = tb.getAttribute('data-sort') !== String(col)+'-asc';
    arr.sort((a,b) => {{
      let x = a.cells[col].innerText, y = b.cells[col].innerText;
      const nx = parseFloat(x), ny = parseFloat(y);
      if (!isNaN(nx) && !isNaN(ny)) {{ x = nx; y = ny; }}
      return (x>y?1:x<y?-1:0) * (asc?1:-1);
    }});
    arr.forEach(r => tb.appendChild(r));
    tb.setAttribute('data-sort', String(col)+(asc?'-asc':'-desc'));
  }}
</script>
</body></html>"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #
def run():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    log_path = os.path.join(OUTPUT_ROOT, "failures.log")
    tuning_cache = {}
    per_sat_rows = {}
    per_sat_rows_all = {}

    def log_failure(name, msg):
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {name}: {msg}\n")

    for name, path_iq, path_dat, path_yml in find_passes(IQ_DIR, DAT_DIR):
        sat_id = sat_id_from_name(name)
        print(f"\n=== {name}  (sat={sat_id}) ===")

        # tuning frequency (cached per satellite)
        try:
            if sat_id not in tuning_cache:
                tuning_cache[sat_id] = read_tuning_freq(path_yml)
                print(f"  loaded f_tune for {sat_id}: {tuning_cache[sat_id]:.0f} Hz")
            f_tune = tuning_cache[sat_id]
        except Exception as e:
            print(f"  yml/tuning read failed -> skip ({e})")
            log_failure(name, f"tuning read failed: {e}")
            continue

        # run the pipeline
        try:
            result = process_pass(
                path_iq, path_dat, f_tune,
                f_s=F_S, N=N, overlap_frac=OVERLAP_FRAC,
                fixed_bw=FIXED_BW, bw_mode=BW_MODE,
                smoother=SMOOTHER, gap_threshold=GAP_THRESHOLD,
                count=COUNT, name=name,
                also_estimate_bw=RUN_TESTS,        # measure band for the bw test
            )
        except Exception as e:
            print(f"  pipeline failed -> skip ({e})")
            log_failure(name, "pipeline:\n" + traceback.format_exc())
            continue

        # output dirs
        sat_dir = os.path.join(OUTPUT_ROOT, sat_id)
        fig_dir = os.path.join(sat_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)

        # pass figure
        try:
            make_pass_figure(result, sat_id, save_path=os.path.join(fig_dir, name + "_pass.png"))
        except Exception as e:
            print(f"  pass figure failed ({e})")
            log_failure(name, f"pass figure: {e}")

        # tests
        test_results = {}
        if RUN_TESTS:
            try:
                test_results["interp"] = check_interpolation_gaps(
                    result["t_ax"], result["f_interp"], result["t_bf"], GAP_THRESHOLD)
                true_bw = KNOWN_BW.get(sat_id)
                test_results["bw"] = check_bandwidth(
                    result["bw_estimated_hz"], true_bw, BW_REL_TOL)
                make_test_figure(result, true_bw_hz=true_bw,
                                 save_path=os.path.join(fig_dir, name + "_test.png"))
            except Exception as e:
                print(f"  tests failed ({e})")
                log_failure(name, f"tests: {e}")

        per_sat_rows.setdefault(sat_id, []).append(
            summary_row(name, sat_id, result, test_results))
        per_sat_rows_all.setdefault("all", []).append(
            summary_row(name, sat_id, result, test_results))

        # flush CSV + HTML after every pass so results are viewable immediately
        sat_dir = os.path.join(OUTPUT_ROOT, sat_id)
        write_csv(os.path.join(sat_dir, "summary.csv"), per_sat_rows[sat_id])
        write_html(os.path.join(sat_dir, "summary.html"), sat_id, per_sat_rows[sat_id])
        write_html(os.path.join(OUTPUT_ROOT, "results_summary.html"), "all", per_sat_rows_all["all"])
        print(f"  done.")

    print("\nbatch complete.")

    


if __name__ == "__main__":
    run()
