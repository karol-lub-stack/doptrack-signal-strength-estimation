import numpy as np
import yaml


def read_data(path_iq, path_dat, dtype=np.complex64, count=-1):
    """Read IQ samples and the best-fit Doppler curve from disk.

    Returns (sig, t_bf, f_bf) or None on failure.
      sig  : complex IQ samples.
      t_bf : best-fit timestamps [s].
      f_bf : best-fit frequencies [Hz], absolute (RF).
    """
    try:
        sig = np.fromfile(path_iq, dtype=dtype, count=count)
        print(f"Read IQ data from {path_iq}")

        dat = np.loadtxt(path_dat, delimiter=None, skiprows=1)
        t_bf = dat[:, 0]
        f_bf = dat[:, 1]
        print(f"Read line data from {path_dat}")

        return sig, t_bf, f_bf
    except Exception as e:
        print(f"Error reading data: {e}")
        return None


def _find_key(obj, *needles):
    """Recursively search a parsed-YAML structure for the first scalar value
    whose key contains ALL given needles (case-insensitive). Returns the value
    or None. Handles nested dicts and lists.
    """
    needles = [n.lower() for n in needles]
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k).lower()
            if all(n in key for n in needles) and not isinstance(v, (dict, list)):
                return v
        for v in obj.values():                       # recurse into children
            found = _find_key(v, *needles)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_key(v, *needles)
            if found is not None:
                return found
    return None


def read_tuning_freq(path_yml):
    """Read the receiver tuning frequency [Hz] from a DopTrack .yml metadata file.

    The DopTrack metadata schema can vary slightly between recordings, so this
    searches for a key containing both 'tuning' and 'frequency'. If your files
    use a different key, adjust the needles below (or the YML structure).

    Returns the frequency as a float, or raises ValueError if not found.
    """
    with open(path_yml, "r", encoding="utf-8") as fh:
        meta = yaml.safe_load(fh)

    val = _find_key(meta, "tuning", "frequency")
    if val is None:
        val = _find_key(meta, "tuning")            # looser fallback
    if val is None:
        raise ValueError(
            f"Could not find a tuning-frequency key in {path_yml}. "
            "Inspect the file and adjust read_tuning_freq's search keys.")
    return float(val)
