import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from StraightenedSignal import straighten_signal
from WaterfallCurveData import update_waterfall_curve
import psutil
#from MovingAverage import moving_average
# Data reading
data = np.fromfile(r"C:\Users\novac\OneDrive\Desktop\Y2 Books\Q3\Project Q3\Data\FUNcube-1_39444_202601010247.fc32", dtype=np.complex64) # Data from L0 Products, which has the IQ data, already downsampled to 25 kHz

# Sampling parameters - Different for each satellite
sampling_rate = 25000  # HZ, ALREADY DOWNSAMPLED
dt = 1 / sampling_rate # s, sampling time
duration = 660 #s
baud_rate = 1200

# Processing parameters - Different for each satellite
chunk_size  = 1200         # controls how many samples we process at once (bigger = better frequency resolution, but more memory)
max_tau = 256 * 2          # maximum lag in samples for WVD calculation. controls the frequency resolution.
stride = chunk_size - 2*max_tau   # overlap between chunks to ensure we capture all time points without edge effects - TO NOT LOSE ENERGY


# Actual WVD calculation function
def wigner_ville_distribution(x, max_tau):
    # instead of looping over each time point t, we build matrices of indices to compute all t+tau and t-tau combinations at once, then do a single FFT across the lag dimension for all time points simultaneously
    N  = len(x)
    taus = np.arange(-max_tau, max_tau + 1)       # lag vector, shape (2*max_tau+1,)
    t_idx = np.arange(max_tau, N - max_tau)       # time indeces, shape (n_t,)
    plus_idx  = t_idx[:, None] + taus[None, :]    # all t+tau combinations,  shape:  (n_t, 2*max_tau+1)
    minus_idx = t_idx[:, None] - taus[None, :]    # all t-tau combinations, shape:  (n_t, 2*max_tau+1)
    R = x[plus_idx] * np.conj(x[minus_idx])       # compute the full autocorrelation matrix, shape: (n_t, 2*max_tau+1)
    # Apply Hanning window across lag axis (axis=1) to suppress cross-terms
    window = np.hanning(2 * max_tau + 1).astype(np.float32)
    R = R * window[None, :]                        # broadcast correctly over all time points
    wvd = np.fft.fftshift(np.fft.fft(R, axis=1), axes=1)
    return np.real(wvd).astype(np.float32)  # cast to float32 here to halve memory

# Compute total chunks for progress display
total_chunks = (len(data) - chunk_size) // stride + 1
t_chunk = chunk_size / sampling_rate  # duration of chunk in seconds
print(f"Total chunks to process: {total_chunks}")
print(f"Each chunk covers {t_chunk:.2f}s of signal")
print("Starting...")

#Initialize C/N lists
P_x_list = []  # will have N_chunks-1 entries, each of shape (N_freq,)
P_s_list = []
prev_wvd = None
n_pairs = 0

# Initialize list
cnr_per_chunk = []

for i in range(total_chunks):
    start = i * stride
    chunk     = data[start : start + chunk_size]
    wvd_chunk = wigner_ville_distribution(chunk, max_tau=max_tau)
    wvd_chunk = gaussian_filter1d(wvd_chunk, sigma=3, axis=1)  # smooth along time axis to reduce noise

    # SNR calculation
    delta_f = sampling_rate / (2 * max_tau + 1)

    wvd_chunk_pos = np.clip(wvd_chunk, 0, None)  # limits the values in an array to a specified range
    # Collapse time into vector: one power value per frequency bin
    power_vs_freq = np.mean(wvd_chunk_pos, axis=0)      # shape: (n_freq,)

    # Signal = peak, Noise = 10th percentile across frequency
    signal = np.max(power_vs_freq)
    noise  = np.percentile(power_vs_freq, 50)

     # Guard: if noise <= 0 after clipping, chunk is all cross-terms, skip it
    if noise <= 0 or signal <= noise:
        cnr_per_chunk.append(np.nan)
    else:
        cnr_linear = signal / noise
        cnr_db     = 10 * np.log10(cnr_linear)
        cnr_per_chunk.append(cnr_db)

    if i % 50 == 0:  # print every 50 chunks, not every chunk
        print(f"  chunk {i}/{total_chunks}  ({100*i/total_chunks:.1f}%) ")
        print(f"Available RAM: {psutil.virtual_memory().available / 1e9:.1f} GB")

cnr_per_chunk = np.array(cnr_per_chunk)
np.savetxt("cnr_per_chunk.csv", cnr_per_chunk, delimiter=",")

# plot C/N over time
time_CNR = np.linspace(0, duration, len(cnr_per_chunk))
plt.figure(figsize=(12, 4))
plt.plot(time_CNR, cnr_per_chunk)
plt.xlabel("Time (s)")
plt.ylabel("Carrier to Noise Ratio (dB)")
plt.title("Carrier to Noise Ratio vs Time")
plt.grid(True)
plt.tight_layout()
plt.show()
