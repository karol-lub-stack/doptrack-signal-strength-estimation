"""Alignment diagnostics.

diagnose_alignment() reports why a carrier (Doppler) track may not line up with
the STFT axes -- the two classic failures being (a) the carrier track and the
STFT use a different time origin/units, or (b) the carrier is in absolute RF Hz
while the STFT axis is baseband (centred on 0). It prints a readable report and
returns a dict of the findings. Handy when a new dataset produces an all-NaN
SNR curve.
"""

import numpy as np


def diagnose_alignment(t_stft, f_stft, t_carrier, f_carrier, center_freq=None,
                       verbose=True):
    t_stft = np.asarray(t_stft, float)
    f_stft = np.asarray(f_stft, float)
    tc = np.asarray(t_carrier)
    fc = np.asarray(f_carrier, float)
    if center_freq is not None:
        fc = fc - center_freq

    is_dt = np.issubdtype(tc.dtype, np.datetime64)
    tcf = (tc.astype("datetime64[ns]").astype(np.int64) / 1e9) if is_dt else tc.astype(float)
    time_overlap = min(t_stft.max(), tcf.max()) - max(t_stft.min(), tcf.min())
    freq_overlap = min(f_stft.max(), fc.max()) - max(f_stft.min(), fc.min())

    span = max(abs(f_stft.min()), abs(f_stft.max()))
    far = np.median(np.abs(fc)) > 5 * span

    rep = dict(
        t_stft_range=(float(t_stft.min()), float(t_stft.max())),
        t_carrier_range=(float(tcf.min()), float(tcf.max())),
        time_overlaps=bool(time_overlap > 0),
        carrier_is_datetime=bool(is_dt),
        f_stft_range=(float(f_stft.min()), float(f_stft.max())),
        f_carrier_range=(float(fc.min()), float(fc.max())),
        freq_overlaps=bool(freq_overlap > 0),
        carrier_looks_absolute_rf=bool(far and center_freq is None),
        suggested_center_freq=(float(np.median(np.asarray(f_carrier, float)))
                               if (far and center_freq is None) else center_freq),
    )

    if verbose:
        print("-- alignment diagnosis ----------------------------------")
        print(f"time  STFT    : [{rep['t_stft_range'][0]:.3f}, {rep['t_stft_range'][1]:.3f}] s")
        print(f"time  carrier : [{rep['t_carrier_range'][0]:.3f}, {rep['t_carrier_range'][1]:.3f}]"
              f"{'  <- datetime, not seconds!' if is_dt else ''}")
        print(f"      overlap : {'yes' if rep['time_overlaps'] else 'NO  -> different origin/units'}")
        print(f"freq  STFT    : [{rep['f_stft_range'][0]:.1f}, {rep['f_stft_range'][1]:.1f}] Hz")
        print(f"freq  carrier : [{rep['f_carrier_range'][0]:.1f}, {rep['f_carrier_range'][1]:.1f}] Hz"
              f"{'  <- looks like absolute RF' if rep['carrier_looks_absolute_rf'] else ''}")
        print(f"      overlap : {'yes' if rep['freq_overlaps'] else 'NO'}")
        if rep["carrier_looks_absolute_rf"]:
            print(f"  fix: subtract center_freq~{rep['suggested_center_freq']:.0f} "
                  "(carrier is RF; f_stft is baseband)")
        if not rep["time_overlaps"]:
            print("  fix: give t_carrier and t_stft the SAME origin "
                  "(seconds from recording start)")
        print("---------------------------------------------------------")
    return rep
