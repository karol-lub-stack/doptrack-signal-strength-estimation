# DopTrack — Signal Strength Estimation

Estimating the received **signal strength (SNR)** of Doppler-shifted CubeSat
signals recorded by TU Delft's [DopTrack](https://doptrack.tudelft.nl/) ground
station.

Three independent implementations are collected here, each in its own subfolder:

| Folder | Method | Author |
|--------|--------|--------|
| `classical/` | Short-Time Fourier Transform (STFT) spectrogram estimator | — |
| `wvd/`       | Smoothed Pseudo Wigner-Ville Distribution (SPWVD) | novaciustina |
| `sst/`       | Synchrosqueezing Transform (SST) | *(coming soon)* |

📊 **[Live results gallery →](https://karol-lub-stack.github.io/doptrack-signal-strength-estimation/)**
— 255 processed passes across four satellites (Delfi-C3, FUNcube-1, Nayif-1,
Delfi-n3Xt), each with its spectrogram, SNR curve and metrics.

---

## Classical STFT estimator (`classical/`)

### How it works

1. **STFT** — the complex baseband recording is transformed with a Hanning-windowed
   short-time Fourier transform (`N = 4096`, 50 % overlap, `fs = 25 kHz`).
2. **Carrier tracking** — the DopTrack `.dat` carrier track is shifted to baseband
   and interpolated onto the STFT time grid; gaps longer than 7 s are detected and
   masked so SNR is only reported where the carrier is reliably known.
3. **Bandwidth** — the occupied signal band is estimated once from a set of
   high-SNR frames (walk-out from the carrier with entry/exit thresholds).
4. **Noise floor** — estimated **per frame** as the median of the out-of-band bins,
   debiased by `ln 2` (exponential noise statistics).
5. **SNR** — in-band excess power over noise, integrated across the band per frame.
6. **Smoothing** — median filter + zero-phase **Butterworth** low-pass.

### Quick start

```bash
pip install -r requirements.txt
```

Edit the config block in [`classical/run_pass.py`](classical/run_pass.py) (satellite,
year, pass id, data paths) and run:

```bash
python classical/run_pass.py
```

### Batch

```bash
python tools/Batch.py
```

### Pipeline parameters

| Parameter      | Default        | Notes                                   |
|----------------|----------------|-----------------------------------------|
| `N`            | `4096`         | STFT frame size                         |
| `overlap_frac` | `0.5`          | 50 % frame overlap                      |
| `f_s`          | `25 000` Hz    | sample rate                             |
| `smoother`     | `butterworth`  | `savgol`, `butterworth`, or `None`      |
| `gap_threshold`| `7.0` s        | carrier-track gap masking threshold     |
| `bw_mode`      | `fixed`        | `fixed` (1200 Hz) or `estimate`         |

---

## WVD estimator (`wvd/`)

Smoothed Pseudo Wigner-Ville Distribution implementation.

**Files included:**

| File | Purpose |
|------|---------|
| `SPWVD.py` | Main WVD computation + HDF5 disk streaming + spectrogram plot |
| `Percentile_snr.py` | Per-chunk SNR via percentile noise floor |
| `StraightenedSignal.py` | Doppler-shift baseband correction (`straighten_signal`) |
| `Low_pass_first_order.py` | First-order IIR low-pass smoother for the SNR curve |
| `MovingAverage.py` | Simple moving-average smoother |

> **Note:** `WaterfallCurveData.py` (provides `update_waterfall_curve`) is a
> dependency referenced by `SPWVD.py`, `Percentile_snr.py`, and
> `StraightenedSignal.py`. Add it to `wvd/` when available.

---

## Repository layout

```
.
+-- classical/           STFT estimator
|   +-- pipeline/          core modules (io, stft, interpolation, bandwidth, snr, smoothing, ...)
|   +-- viz/               figures.py, tests.py
|   +-- run_pass.py        single-pass entry point
+-- wvd/                 SPWVD estimator
|   +-- SPWVD.py
|   +-- Percentile_snr.py
|   +-- StraightenedSignal.py
|   +-- Low_pass_first_order.py
|   +-- MovingAverage.py
+-- sst/                 SST estimator (coming soon)
+-- tools/               shared batch tools & helpers
|   +-- Batch.py
|   +-- Batch2.py
|   +-- build_gallery.py
|   +-- extract_snr_curves.py
|   +-- AIBW.py
|   +-- analysis.py
+-- docs/                GitHub Pages results gallery (static site)
```

> Raw IQ data (`.fc32`), carrier tracks (`.dat`) and local batch outputs are **not**
> committed — see `.gitignore`. Point the config paths at your own DopTrack data.

---

## Rebuild the results gallery

After a batch run:

```bash
python tools/build_gallery.py
```

This web-optimises the figures and regenerates `docs/` (served by GitHub Pages).
