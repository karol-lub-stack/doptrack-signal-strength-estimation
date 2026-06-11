import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from StraightenedSignal import straighten_signal
from WaterfallCurveData import update_waterfall_curve
#from MovingAverage import moving_average
# Data reading
#data = np.fromfile(r"C:\Users\novac\OneDrive\Desktop\Y2 Books\Q3\Project Q3\Data\FUNcube-1_39444_202601010247.fc32", dtype=np.complex64) # Data from L0 Products, which has the IQ data, already downsampled to 25 kHz
data = np.fromfile(r"C:\Users\novac\Downloads\FUNcube-1_39444_202601010421.fc32", dtype=np.complex64)
waterfall_data = np.loadtxt('C:\\Users\\novac\\Downloads\\FUNcube-1_39444_202601010247.dat', skiprows=1) # Data from L1B Products, which has the waterfall curve extracted

# Sampling parameters - Diffrent for each satellite
sampling_rate = 25000  # HZ, ALREADY DOWNSAMPLED
dt = 1 / sampling_rate # s, sampling time
#duration = 660 #s
duration = 840
baud_rate = 1200

# Processing parameters - Diffrent for each satellite
chunk_size  = 1200         # controls how many samples we process at once (bigger = better frequency resolution, but more memory)
max_tau = 256  * 2       # maximum lag in samples for WVD calculation. controls the frequency resolution.
stride = chunk_size - 2*max_tau   # overlap between chunks to ensure we capture all time points without edge effects - TO NOT LOSE ENERGY

# Power lists
power_time = []   # instantaneous power over time
power_time_updated = []
time_power_updated = []
time_power = []   # corresponding time points for power plot
total_energy = 0  # total energy
"""
#Preprocessing

data = straighten_signal(data, duration, waterfall_data)  # shift the signal to 0 Hz before processing

# Low pass filter function
def low_pass_filter(signal, cutoff_freq, sampling_rate):
    from scipy.signal import butter, filtfilt
    # butter function returns filter coefficients for a Butterworth low-pass filter
    # filtfilt applies the filter forward and backward to avoid phase distortion, giving zero-phase filtering
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    N = 4  # filter order, higher = sharper cutoff but more ringing
    b, a = butter(N, normal_cutoff, btype='low', analog=False)
    filtered_signal = filtfilt(b, a, signal) # deals with edge effects by applying the filter in both directions
    return filtered_signal



# Apply low pass filter to the entire signal before processing to reduce high-frequency noise that can cause artifacts in the WVD
cutoff_freq = bandwidth = baud_rate *0.5  # cutoff frequency for low-pass filter, set to 70% of the baud rate to preserve the main signal while reducing noise
data = low_pass_filter(data, cutoff_freq=cutoff_freq, sampling_rate=sampling_rate)

"""
# Actual WVD calculation function
def wigner_ville_distribution(x, max_tau):
    # instead of looping over each time point t, we build matrices of indices to compute all t+tau and t-tau combinations at once, then do a single FFT across the lag dimension for all time points simultaneously
    N  = len(x)
    taus = np.arange(-max_tau, max_tau + 1)       # lag vector, shape (2*max_tau+1,)
    t_idx = np.arange(max_tau, N - max_tau)       # time indeces, shape (n_t,)
    plus_idx  = t_idx[:, None] + taus[None, :]    # all t+tau combinations,  shape:  (n_t, 2*max_tau+1)
    minus_idx = t_idx[:, None] - taus[None, :]    # all t-tau combinations, shape:  (n_t, 2*max_tau+1)
    R = x[plus_idx] * np.conj(x[minus_idx])       # compute the full autocorrelation matrix, shape: (n_t, 2*max_tau+1)
    # A Hanning window can be applied in the lag domain
    # it indeed smoothens the WVD and reduces cross-term artifacts
    #window = np.hanning(2 * max_tau + 1).astype(np.float32)
    #R = R * window[None, :]
    wvd = np.fft.fftshift(np.fft.fft(R, axis=1), axes=1)
    return np.real(wvd).astype(np.float32)  # cast to float32 here to halve memory

# Compute total chunks for progress display
# total_chunks = len(data) // chunk_size
total_chunks = (len(data) - chunk_size) // stride + 1
t_chunk = chunk_size / sampling_rate  # duration of chunk in seconds
print(f"Total chunks to process: {total_chunks}")
print(f"Each chunk covers {t_chunk:.2f}s of signal")
print("Starting...")


#Initialize SNR lists
P_x_list = []  # will have N_chunks-1 entries, each of shape (N_freq,)
P_s_list = []
prev_wvd_mean = None

#Disk memory allocation
with h5py.File("wvd_output.h5", "w") as f:
    dset = f.create_dataset( "wvd", shape=(0, 2 * max_tau + 1), maxshape=(None, 2 * max_tau + 1), dtype=np.float32, chunks=(256, 2 * max_tau + 1)  )
    row = 0  # keep track of how many rows we've written to disk so far
    # compute WVD in chunks and write each chunk to disk immediately to avoid RAM issues
    #for i, start in enumerate(range(0, len(data) - chunk_size, stride * stride_skip)):
    for i in range(total_chunks):
        start = i * stride
        chunk     = data[start : start + chunk_size]
        wvd_chunk = wigner_ville_distribution(chunk, max_tau=max_tau)
        wvd_chunk = gaussian_filter1d(wvd_chunk, sigma=3, axis=0)  # smooth along time axis to reduce noise
        # Sigma controls the amount of smoothing, higher = smoother but more blurring
        # Instantaneous power (sum over frequency axis)
        # p_chunk = np.sum(wvd_chunk, axis=1)
        # p_chunk_updated = np.sum(wvd_chunk) / t_chunk
        # power_time_updated.append(p_chunk_updated)
        # power_time.append(p_chunk)
        # # Total energy
        # total_energy += np.sum(wvd_chunk)
        new_row = row + len(wvd_chunk)
        dset.resize(new_row, axis=0)       # grow disk dataset by this chunk's rows
        dset[row:new_row] = wvd_chunk      # write to disk — chunk freed after this line
        row = new_row  # update our row counter to the new total on disk
        # print every 50 chunks
        if i % 50 == 0:  # print every 50 chunks, not every chunk
            print(f"  chunk {i}/{total_chunks}  ({100*i/total_chunks:.1f}%)  --  {row} rows on disk")

    print(f"Done. Reading {row} rows back from disk for plotting...")
    #wvd = dset[:]  # read back from disk into RAM for plotting (now that all processing is done)

# Combine power and time lists into single array for plotting
# power_time_updated = np.array(power_time_updated)
# time_power_updated = np.linspace(0, duration, total_chunks)
# power_time = np.concatenate(power_time)
# time_power = np.linspace(0, duration, len(power_time))

# np.savetxt("Power_list_updated.csv", power_time_updated, delimiter=",")


#power_time_updated = moving_average(power_time_updated, window_size=10)  # smooth the power over time with a moving average to reduce noise in the power plot
"""
# Plotting
num_time = wvd.shape[0]
num_freq = wvd.shape[1]

# Visualisation
max_display_rows = 2000                                    # target pixel height
block_size  = max(1, num_time // max_display_rows)         # rows to average together
trimmed     = num_time - (num_time % block_size)           #  trim to exact multiple
wvd_display = wvd[:trimmed].reshape(-1, block_size, num_freq).mean(axis=1)  # average down
print(f"Display shape: {wvd_display.shape}  (downsampled from {num_time} rows)")
"""

# read in blocks for plotting
max_display_rows = 2000

with h5py.File("wvd_output.h5", "r") as f:
    dset = f["wvd"]
    total_rows = dset.shape[0]
    num_freq   = dset.shape[1]
    block_size = max(1, total_rows // max_display_rows)

    wvd_display = np.zeros((max_display_rows, num_freq), dtype=np.float32)
    for i in range(max_display_rows):
        row_start = i * block_size
        row_end   = min(row_start + block_size, total_rows)
        wvd_display[i] = dset[row_start:row_end].mean(axis=0)

print(f"Display shape: {wvd_display.shape}")


# Time and frequency axes
time_axis = np.linspace(0, duration, wvd_display.shape[0])
freqs     = np.fft.fftshift(np.fft.fftfreq(num_freq, d=1 / sampling_rate))

# convert to dB for better visualization
wvd_db = 10 * np.log10(np.abs(wvd_display) + 1e-20)
vmin   = np.percentile(wvd_db, 5)
vmax   = np.percentile(wvd_db, 99)

# Plot the Wigner-Ville distribution
plt.figure(figsize=(14, 7))
plt.imshow(wvd_db, aspect='auto', origin='lower', extent=[freqs[0], freqs[-1], time_axis[0], time_axis[-1]],  vmin=vmin, vmax=vmax)
plt.colorbar(label="Power/Frequency (dBW/Hz)")
plt.gca().invert_yaxis() # invert y-axis to have time increasing downwards
plt.ylabel("Time (s)")
plt.xlabel("Frequency (Hz)")
plt.title("Wigner-Ville Distribution")
plt.tight_layout()
plt.show()

# # Plot power over time
# plt.figure(figsize=(12, 4))
# plt.plot(time_power, 10*np.log10(np.abs(power_time) + 1e-20))
# plt.xlabel("Time (s)")
# plt.ylabel("Power (dBW)")
# plt.title("Instantaneous Power vs Time")
# plt.grid(True)
# plt.tight_layout()
# plt.xlim(0, duration)
# plt.show()

# # Plot updated power over time
# plt.figure(figsize=(12, 4))
# plt.plot(time_power_updated, 10*np.log10(np.abs(power_time_updated) + 1e-20))
# plt.xlabel("Time (s)")
# plt.ylabel("Power (dBW)")
# plt.title("Instantaneous UPDATED Power vs Time")
# plt.grid(True)
# plt.tight_layout()
# plt.xlim(0, duration)
# plt.show()
