"""Comparison batch runner.

Reads pass names from an existing summary.csv (produced by Batch.py), reruns
those same passes under one or more alternative configurations, then writes a
side-by-side comparison HTML so results can be compared visually.

Output layout:
    OUTPUT_ROOT/
        <config_label>/
            <Satellite>/figures/<pass>_pass.png
            <Satellite>/figures/<pass>_pass_snr.png
        comparison.html     <- one row per pass, columns per config

Edit CONFIGS below to define what you want to compare.
SOURCE_CSV points at the existing batch results to replay.
"""

import os
import glob
import csv
import html
import traceback
import datetime as dt

import numpy as np
import matplotlib
matplotlib.use("Agg")

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "classical"))
from pipeline.io import read_tuning_freq
from pipeline.process_pass import process_pass
from viz.figures import make_pass_figure

# --------------------------------------------------------------------------- #
# where the existing results live (used only to get the pass list)
# --------------------------------------------------------------------------- #
SOURCE_ROOT = r"C:\Users\glute\Desktop\Project Python Folder\Data Dopptrack\results\bwtest"
IQ_DIR      = r"X:\data\L0"
DAT_DIR     = r"X:\data\L1B"
OUTPUT_ROOT = r"C:\Users\glute\Desktop\Project Python Folder\Data Dopptrack\results\comparison"

# --------------------------------------------------------------------------- #
# configs to compare — label : pipeline kwargs
# each label becomes a column in the comparison HTML
# --------------------------------------------------------------------------- #
CONFIGS = {
    "baseline (savgol)": dict(
        N=2**12, overlap_frac=0.5, fixed_bw=1200, bw_mode="fixed",
        smoother="savgol", gap_threshold=7.0,
    ),
    "butterworth": dict(
        N=2**12, overlap_frac=0.5, fixed_bw=1200, bw_mode="fixed",
        smoother="butterworth", gap_threshold=7.0,
    ),
    # --- uncomment to add more ---
    # "no smoother": dict(
    #     N=2**12, overlap_frac=0.5, fixed_bw=1200, bw_mode="fixed",
    #     smoother=None, gap_threshold=7.0,
    # ),
    # "N=1024": dict(
    #     N=1024, overlap_frac=0.5, fixed_bw=1200, bw_mode="fixed",
    #     smoother="savgol", gap_threshold=7.0,
    # ),
    # "N=2048": dict(
    #     N=2048, overlap_frac=0.5, fixed_bw=1200, bw_mode="fixed",
    #     smoother="savgol", gap_threshold=7.0,
    # ),
    # "bw=estimate": dict(
    #     N=2**12, overlap_frac=0.5, fixed_bw=1200, bw_mode="estimate",
    #     smoother="savgol", gap_threshold=7.0,
    # ),
    # "overlap=0.75": dict(
    #     N=2**12, overlap_frac=0.75, fixed_bw=1200, bw_mode="fixed",
    #     smoother="savgol", gap_threshold=7.0,
    # ),
}

F_S = 25_000


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_passes_from_csvs(source_root):
    """Read all summary.csvs under source_root and return unique (name, sat) pairs."""
    passes = {}
    for csv_path in glob.glob(os.path.join(source_root, "*", "summary.csv")):
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                name = row["name"]
                sat  = row["satellite"]
                if name not in passes:
                    passes[name] = sat
    return passes  # {name: satellite}


def resolve_paths(name, sat):
    parts = name.split("_")
    year  = parts[2][:4] if len(parts) >= 3 else "unknown"
    path_iq  = os.path.join(IQ_DIR,  sat, year, name + ".fc32")
    path_dat = os.path.join(DAT_DIR, sat, year, name + ".dat")
    path_yml = os.path.join(IQ_DIR,  sat, year, name + ".yml")
    return path_iq, path_dat, path_yml


def log_failure(log_path, name, msg):
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {name}: {msg}\n")


# --------------------------------------------------------------------------- #
# comparison HTML
# --------------------------------------------------------------------------- #
def write_comparison_html(path, configs, rows_by_config, all_names, all_sats):
    """One row per pass, one column-group per config. Clicking a cell loads its figures."""
    labels   = list(configs.keys())
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    n_passes = len(all_names)

    # table header: Pass | Sat | [label: peak_snr, median_snr] ...
    th_pass = '<th>pass</th><th>satellite</th>'
    th_cfg  = "".join(
        f'<th colspan="2" class="cfg-head">{html.escape(l)}<br>'
        f'<span class="subh">peak&nbsp;dB / median&nbsp;dB</span></th>'
        for l in labels
    )

    body_rows = []
    for name in all_names:
        sat = all_sats[name]
        cells = f'<td class="pass-name">{html.escape(name)}</td><td>{html.escape(sat)}</td>'
        for label in labels:
            row = rows_by_config.get(label, {}).get(name)
            if row:
                peak   = row.get("peak_snr_db",   "—")
                median = row.get("median_snr_db",  "—")
                cfg_slug = label.replace(" ", "_").replace("(", "").replace(")", "")
                fig_dir  = f"{cfg_slug}/{sat}/figures"
                cells += (
                    f'<td class="snr-cell" '
                    f'data-spec="{fig_dir}/{html.escape(name)}_pass.png" '
                    f'data-snr="{fig_dir}/{html.escape(name)}_pass_snr.png">'
                    f'{peak}</td>'
                    f'<td class="snr-cell med" '
                    f'data-spec="{fig_dir}/{html.escape(name)}_pass.png" '
                    f'data-snr="{fig_dir}/{html.escape(name)}_pass_snr.png">'
                    f'{median}</td>'
                )
            else:
                cells += '<td class="missing" colspan="2">—</td>'
        body_rows.append(f'<tr data-name="{html.escape(name)}">{cells}</tr>')

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Config comparison — {n_passes} passes</title>
<style>
  :root {{ --bg:#0f1419; --panel:#1a2029; --line:#2a3340; --ink:#d6dde6;
           --accent:#5ec8f2; --muted:#7d8a99; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
          font:13px/1.5 "JetBrains Mono",ui-monospace,monospace; }}
  header {{ padding:14px 20px; border-bottom:1px solid var(--line); }}
  h1 {{ margin:0; font-size:16px; }}
  .sub {{ color:var(--muted); font-size:11px; margin-top:3px; }}
  .wrap {{ display:flex; height:calc(100vh - 52px); }}
  .left {{ flex:1 1 65%; overflow:auto; border-right:1px solid var(--line); }}
  .right {{ flex:0 0 35%; overflow:auto; padding:14px; }}
  table {{ border-collapse:collapse; width:100%; font-size:11.5px; }}
  th,td {{ padding:5px 8px; text-align:left; border-bottom:1px solid var(--line);
           white-space:nowrap; }}
  th {{ position:sticky; top:0; background:var(--panel); color:var(--accent); }}
  .cfg-head {{ border-left:2px solid var(--line); font-size:11px; }}
  .subh {{ color:var(--muted); font-weight:normal; font-size:10px; }}
  .pass-name {{ color:var(--muted); font-size:10.5px; }}
  tbody tr {{ cursor:pointer; }}
  tbody tr:hover {{ background:#161d27; }}
  tbody tr.sel {{ background:#143041; }}
  .snr-cell {{ text-align:right; cursor:pointer; }}
  .snr-cell:hover {{ color:#fff; background:#1e2d3a; }}
  .med {{ color:var(--muted); border-right:2px solid var(--line); }}
  .missing {{ color:#444; text-align:center; border-right:2px solid var(--line); }}
  .fig {{ margin-bottom:14px; }}
  .fig h3 {{ font-size:11px; color:var(--muted); margin:0 0 4px; font-weight:400; }}
  .fig img {{ width:100%; border:1px solid var(--line); border-radius:3px; background:#fff; }}
  .hint {{ color:var(--muted); padding:20px; font-size:12px; }}
  .cfg-label {{ font-size:11px; color:var(--accent); margin-bottom:8px; }}
</style></head>
<body>
<header>
  <h1>Config comparison</h1>
  <div class="sub">{n_passes} passes &middot; {len(labels)} configs &middot; generated {generated}
    &middot; click a cell to view its figures</div>
</header>
<div class="wrap">
  <div class="left">
    <table id="t">
      <thead><tr>{th_pass}{th_cfg}</tr></thead>
      <tbody>{''.join(body_rows)}</tbody>
    </table>
  </div>
  <div class="right" id="panel"><div class="hint">Click any SNR value to view its spectrogram and SNR figure.</div></div>
</div>
<script>
  document.querySelectorAll('.snr-cell').forEach(td => {{
    td.addEventListener('click', e => {{
      e.stopPropagation();
      document.querySelectorAll('tr.sel').forEach(r => r.classList.remove('sel'));
      td.closest('tr').classList.add('sel');
      const spec = td.dataset.spec, snr = td.dataset.snr;
      // find config label from column index
      const idx = td.cellIndex;
      const hdrs = document.querySelectorAll('th.cfg-head');
      let cfg = '';
      let col = 2;
      hdrs.forEach(h => {{ if (col === idx || col+1 === idx) cfg = h.innerText.split('\\n')[0]; col += 2; }});
      document.getElementById('panel').innerHTML =
        '<div class="cfg-label">' + cfg + '</div>'
        + '<div class="fig"><h3>spectrogram</h3>'
        + '<img src="' + spec + '" onerror="this.parentNode.innerHTML=\\'<h3 style=color:#f2776e>figure missing</h3>\\'"></div>'
        + '<div class="fig"><h3>SNR</h3>'
        + '<img src="' + snr  + '" onerror="this.parentNode.style.display=\\'none\\'"></div>';
    }});
  }});
</script>
</body></html>"""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def run():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    log_path = os.path.join(OUTPUT_ROOT, "failures.log")

    print("Loading existing passes from source CSVs...")
    pass_map = load_passes_from_csvs(SOURCE_ROOT)
    print(f"  {len(pass_map)} unique passes found.\n")

    if not pass_map:
        print("No passes found — check SOURCE_ROOT.")
        return

    tuning_cache = {}
    rows_by_config = {label: {} for label in CONFIGS}

    for label, cfg in CONFIGS.items():
        cfg_slug = label.replace(" ", "_").replace("(", "").replace(")", "")
        print(f"\n{'='*60}")
        print(f"CONFIG: {label}")
        print(f"{'='*60}")

        for name, sat in pass_map.items():
            path_iq, path_dat, path_yml = resolve_paths(name, sat)
            if not all(os.path.exists(p) for p in [path_iq, path_dat, path_yml]):
                print(f"  [skip] {name} — files missing")
                log_failure(log_path, name, f"[{label}] source files missing")
                continue

            print(f"  {name} ({sat})", end=" ... ", flush=True)

            try:
                if sat not in tuning_cache:
                    tuning_cache[sat] = read_tuning_freq(path_yml)
                f_tune = tuning_cache[sat]
            except Exception as e:
                print(f"yml failed ({e})")
                log_failure(log_path, name, f"[{label}] tuning: {e}")
                continue

            try:
                result = process_pass(
                    path_iq, path_dat, f_tune,
                    f_s=F_S, name=name,
                    also_estimate_bw=True,
                    **cfg,
                )
            except Exception as e:
                print(f"pipeline failed ({e})")
                log_failure(log_path, name, f"[{label}] pipeline:\n{traceback.format_exc()}")
                continue

            fig_dir  = os.path.join(OUTPUT_ROOT, cfg_slug, sat, "figures")
            os.makedirs(fig_dir, exist_ok=True)
            try:
                make_pass_figure(result, sat_id=sat,
                                 save_path=os.path.join(fig_dir, name + "_pass.png"))
            except Exception as e:
                print(f"figure failed ({e})")
                log_failure(log_path, name, f"[{label}] figure: {e}")

            snr = result["snr_db"]
            finite = np.isfinite(snr)
            rows_by_config[label][name] = dict(
                peak_snr_db   = round(float(np.nanmax(snr)),    2) if finite.any() else "—",
                median_snr_db = round(float(np.nanmedian(snr)), 2) if finite.any() else "—",
            )
            print("done")

        # flush comparison HTML after each config so it's viewable mid-run
        all_names = list(pass_map.keys())
        write_comparison_html(
            os.path.join(OUTPUT_ROOT, "comparison.html"),
            CONFIGS, rows_by_config, all_names, pass_map,
        )

    print(f"\nAll done. Open: {os.path.join(OUTPUT_ROOT, 'comparison.html')}")


if __name__ == "__main__":
    run()
