# DopTrack — Signal Strength Estimation

Estimating the received **signal strength (SNR)** of Doppler-shifted CubeSat
signals recorded by TU Delft's [DopTrack](https://doptrack.tudelft.nl/) ground
station, using a **classical spectrogram** estimator.

The pipeline takes a raw IQ recording plus the DopTrack carrier track, builds a
short-time Fourier spectrogram, follows the Doppler carrier across the pass, and
reports a per-frame in-band SNR curve.

📊 **[Live results gallery →](https://karol-lub-stack.github.io/doptrack-signal-strength-estimation/)**
— 255 processed passes across four satellites (Delfi-C3, FUNcube-1, Nayif-1,
Delfi-n3Xt), each with its spectrogram, SNR curve and metrics.

---

## How it works

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

---

## Repository layout

```
.
├── pipeline/            core single-pass estimator
│   ├── io.py              read IQ + carrier track, tuning frequency
│   ├── stft.py            short-time Fourier transform
│   ├── interpolation.py   carrier-track interpolation + gap masking
│   ├── bandwidth.py       occupied-bandwidth estimation
│   ├── snr.py             per-frame noise floor + SNR
│   ├── smoothing.py       Savitzky-Golay / Butterworth smoothers
│   ├── power.py           power helpers
│   ├── diagnostics.py     alignment checks
│   └── process_pass.py    orchestrates one pass → result dict
├── viz/                 plotting
│   ├── figures.py         spectrogram + SNR figures
│   └── tests.py           interpolation / bandwidth diagnostics
├── run_pass.py          ← run the pipeline on a single pass
├── tools/               batch processing & helpers
│   ├── Batch.py           process every recording in a data tree
│   ├── Batch2.py          re-run passes under multiple configs + compare
│   ├── build_gallery.py   build the docs/ results gallery
│   ├── extract_snr_curves.py
│   ├── AIBW.py            bandwidth-parameter optimiser
│   └── analysis.py        pipeline benchmarking
└── docs/                GitHub Pages results gallery (static site)
```

> Raw IQ data (`.fc32`), carrier tracks (`.dat`) and local batch outputs are **not**
> committed — see `.gitignore`. Point the config paths at your own DopTrack data.

---

## Quick start

```bash
pip install -r requirements.txt
```

### Single pass

Edit the config block in [`run_pass.py`](run_pass.py) (satellite, year, pass id,
data paths) and run:

```bash
python run_pass.py
```

This prints the bandwidth and peak/median SNR and shows the spectrogram + SNR figure.

### Batch

Edit the paths and parameters at the top of [`tools/Batch.py`](tools/Batch.py),
then:

```bash
python tools/Batch.py
```

It writes per-satellite figures, a `summary.csv`, and a live-updating HTML summary.

### Rebuild the results gallery

After a batch run:

```bash
python tools/build_gallery.py
```

This web-optimises the figures and regenerates `docs/` (served by GitHub Pages).

---

## Pipeline parameters

| Parameter      | Default        | Notes                                   |
|----------------|----------------|-----------------------------------------|
| `N`            | `4096`         | STFT frame size                         |
| `overlap_frac` | `0.5`          | 50 % frame overlap                      |
| `f_s`          | `25 000` Hz    | sample rate                             |
| `smoother`     | `butterworth`  | `savgol`, `butterworth`, or `None`      |
| `gap_threshold`| `7.0` s        | carrier-track gap masking threshold     |
| `bw_mode`      | `fixed`        | `fixed` (1200 Hz) or `estimate`         |
