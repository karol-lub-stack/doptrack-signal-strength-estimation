"""Pre-compute and save STFT spectrograms to disk as compressed .npz files.

Walks L0/<satellite>/<year>/*.fc32 and writes:
    OUTPUT_ROOT/<satellite>/<year>/<name>.npz   (t_ax, f_ax, S)
"""

import os
import glob

import numpy as np

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "classical"))
from pipeline.stft import stft


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
IQ_DIR = r"X:\data\L0"
OUTPUT_ROOT = r"X:\data\STFT"

F_S = 25_000
N = 2 ** 12
OVERLAP_FRAC = 0.5
COUNT = -1


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sat_id_from_name(name):
    return name.split("_")[0]


def read_iq(path_iq, dtype=np.complex64, count=-1):
    try:
        sig = np.fromfile(path_iq, dtype=dtype, count=count)
        print(f"Read IQ data from {path_iq}")
        return sig
    except Exception as e:
        print(f"Error reading data: {e}")
        return None


def _process_pass(path_iq, f_s=25_000, N=2 ** 12, overlap_frac=0.5, count=-1):
    N = int(N)
    hop = int(N * (1 - overlap_frac))
    sig = read_iq(path_iq, count=count)
    if sig is None:
        raise RuntimeError(f"failed to read {path_iq}")
    t_ax, f_ax, S = stft(sig, N, hop, f_s, window=np.hanning(N))
    return dict(t_ax=t_ax, f_ax=f_ax, S=S)


def find_passes(iq_dir):
    candidates = []
    for path_iq in sorted(glob.glob(os.path.join(iq_dir, "*", "*", "*.fc32"))):
        name = os.path.splitext(os.path.basename(path_iq))[0]
        rel = os.path.relpath(path_iq, iq_dir)
        parts = rel.split(os.sep)
        sat_year = os.path.join(*parts[:-1])
        candidates.append((name, path_iq, sat_year))
    print(f"Found {len(candidates)} passes")
    return candidates


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #
def run():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for name, path_iq, sat_year in find_passes(IQ_DIR):
        sat_id = sat_id_from_name(name)
        print(f"\n=== {name}  (sat={sat_id}) ===")
        result = _process_pass(path_iq, f_s=F_S, N=N, overlap_frac=OVERLAP_FRAC, count=COUNT)

        sat_dir = os.path.join(OUTPUT_ROOT, sat_id, sat_year)
        os.makedirs(sat_dir, exist_ok=True)

        outpath = os.path.join(sat_dir, name)
        np.savez_compressed(outpath, t_ax=result["t_ax"], f_ax=result["f_ax"], S=result["S"])
        print(f"  Saved -> {outpath}.npz")

    print("\nbatch_stft complete.")


if __name__ == "__main__":
    run()
