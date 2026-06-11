"""Build the GitHub Pages results gallery.

Reads the per-satellite summary.csv files produced by tools/Batch.py, copies and
web-optimises each pass figure into docs/ (a small thumbnail for the grid plus a
larger image for the modal), and emits docs/results.json that drives the viewer.

Run locally after a batch:  python tools/build_gallery.py
The docs/ output is what gets committed and served by GitHub Pages.
"""

import os
import csv
import json
import datetime as dt
from statistics import median, mean

from PIL import Image

# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
RESULTS_ROOT = r"C:\Users\glute\Desktop\Project Python Folder\Data Dopptrack\results\bwtest"
REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS         = os.path.join(REPO_ROOT, "docs")

THUMB_W = 460     # grid thumbnail width (px)
FULL_W  = 1280    # modal image width (px)
THUMB_Q = 80
FULL_Q  = 88

SAT_ORDER = ["Delfi-C3", "FUNcube-1", "Nayif-1", "Delfi-n3Xt"]

# columns we surface in the viewer (csv name -> nice label)
NUMERIC = ["peak_snr_db", "median_snr_db", "duration_s", "n_frames",
           "bw_used_hz", "bw_estimated_hz", "interp_n_gaps", "bw_rel_error"]


# --------------------------------------------------------------------------- #
def optimise(src_png, dst_jpg, width, quality):
    """Resize (preserving aspect) to `width` and save as optimised JPEG."""
    os.makedirs(os.path.dirname(dst_jpg), exist_ok=True)
    with Image.open(src_png) as im:
        im = im.convert("RGB")
        if im.width > width:
            h = round(im.height * width / im.width)
            im = im.resize((width, h), Image.LANCZOS)
        im.save(dst_jpg, "JPEG", quality=quality, optimize=True, progressive=True)
    return os.path.getsize(dst_jpg)


def fnum(v):
    try:
        f = float(v)
        return f if f == f else None      # drop NaN
    except (TypeError, ValueError):
        return None


def stat_block(rows, key):
    vals = [fnum(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return dict(min=round(min(vals), 2), max=round(max(vals), 2),
                median=round(median(vals), 2), mean=round(mean(vals), 2))


def pass_rate(rows, key):
    truthy = [str(r.get(key, "")).strip().lower() in ("true", "1", "yes") for r in rows
              if str(r.get(key, "")).strip() != ""]
    if not truthy:
        return None
    return round(100 * sum(truthy) / len(truthy), 1)


# --------------------------------------------------------------------------- #
def main():
    if not os.path.isdir(RESULTS_ROOT):
        raise SystemExit(f"results root not found: {RESULTS_ROOT}")

    sats_found = [d for d in os.listdir(RESULTS_ROOT)
                  if os.path.isfile(os.path.join(RESULTS_ROOT, d, "summary.csv"))]
    sats = [s for s in SAT_ORDER if s in sats_found] + \
           [s for s in sorted(sats_found) if s not in SAT_ORDER]

    total_bytes = 0
    payload_sats = []
    n_total = 0

    for sat in sats:
        csv_path = os.path.join(RESULTS_ROOT, sat, "summary.csv")
        with open(csv_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))

        kept = []
        for r in rows:
            name = r.get("name", "").strip()
            if not name:
                continue
            src = os.path.join(RESULTS_ROOT, sat, "figures", name + "_pass.png")
            if not os.path.exists(src):
                continue

            thumb = os.path.join(DOCS, "thumbs",  sat, name + ".jpg")
            full  = os.path.join(DOCS, "figures", sat, name + ".jpg")
            total_bytes += optimise(src, thumb, THUMB_W, THUMB_Q)
            total_bytes += optimise(src, full,  FULL_W,  FULL_Q)

            # parse a date from the pass id  <sat>_<id>_YYYYMMDDhhmm
            date = ""
            parts = name.split("_")
            if len(parts) >= 3 and len(parts[2]) >= 8:
                d = parts[2]
                date = f"{d[0:4]}-{d[4:6]}-{d[6:8]} {d[8:10]}:{d[10:12]}"

            kept.append(dict(
                name=name, date=date,
                peak_snr_db=fnum(r.get("peak_snr_db")),
                median_snr_db=fnum(r.get("median_snr_db")),
                duration_s=fnum(r.get("duration_s")),
                n_frames=fnum(r.get("n_frames")),
                bw_mode=r.get("bw_mode", ""),
                bw_used_hz=fnum(r.get("bw_used_hz")),
                bw_estimated_hz=fnum(r.get("bw_estimated_hz")),
                interp_ok=str(r.get("interp_ok", "")).strip().lower() in ("true", "1", "yes"),
                interp_n_gaps=fnum(r.get("interp_n_gaps")),
                bw_test_ok=str(r.get("bw_test_ok", "")).strip().lower() in ("true", "1", "yes"),
                bw_rel_error=fnum(r.get("bw_rel_error")),
                thumb=f"thumbs/{sat}/{name}.jpg",
                full=f"figures/{sat}/{name}.jpg",
            ))

        n_total += len(kept)
        payload_sats.append(dict(
            name=sat,
            count=len(kept),
            stats=dict(
                peak_snr_db=stat_block(kept, "peak_snr_db"),
                median_snr_db=stat_block(kept, "median_snr_db"),
                duration_s=stat_block(kept, "duration_s"),
                bw_estimated_hz=stat_block(kept, "bw_estimated_hz"),
                interp_pass_pct=pass_rate(kept, "interp_ok"),
                bw_test_pass_pct=pass_rate(kept, "bw_test_ok"),
            ),
            passes=kept,
        ))
        print(f"  {sat:<12} {len(kept):>4} passes")

    out = dict(
        generated=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        n_passes=n_total,
        n_satellites=len(payload_sats),
        config=dict(smoother="butterworth", N=4096, overlap=0.5,
                    fs_hz=25000, gap_threshold_s=7.0),
        satellites=payload_sats,
    )
    os.makedirs(DOCS, exist_ok=True)
    blob = json.dumps(out, separators=(",", ":"))
    # JSON for tooling, plus a JS shim so the viewer also works opened locally
    with open(os.path.join(DOCS, "results.json"), "w", encoding="utf-8") as fh:
        fh.write(blob)
    with open(os.path.join(DOCS, "data.js"), "w", encoding="utf-8") as fh:
        fh.write("window.RESULTS = " + blob + ";")

    print(f"\n  {n_total} passes across {len(payload_sats)} satellites")
    print(f"  optimised image payload: {total_bytes/1e6:.1f} MB")
    print(f"  wrote results.json + data.js")


if __name__ == "__main__":
    main()
