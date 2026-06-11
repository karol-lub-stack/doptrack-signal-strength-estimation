import numpy as np
import matplotlib.pyplot as plt

delta_t = 0.05 #chunck time
duration = 660 #total duration of signal


x = np.loadtxt("cnr_per_chunk.csv", delimiter=",")
#x = 10 * np.log10(x + 1e-20)  # convert to dB, add small value to avoid log of zero
SNR = x.copy()

cutoff_freq = delta_t/5
omega = 2 * np.pi * cutoff_freq
def low_pass_filter(delta_t, omega, x):
    y = x
    alpha = ( 2- delta_t * omega ) / ( 2 + delta_t * omega )
    beta = delta_t * omega / ( 2 + delta_t * omega )
    for k in range(1, len(x)):
        y[k] = alpha * y[k-1] + beta * ( x[k] + x[k-1] )
    return y

SNR_Smoothed = low_pass_filter(delta_t, omega, x)
time = np.linspace(0, duration, len(x))
single_value = np.mean(SNR_Smoothed)
print(f"Single SNR value for the whole signal: {single_value:.2f} dB")
plt.plot(time, SNR, color='lightgray', alpha=0.5, label='Raw SNR')
plt.plot(time, SNR_Smoothed, color='red', linewidth=2, label='Smoothed SNR')
plt.title('SNR Estimation')
plt.xlabel('Time (s)')
plt.ylabel('SNR (dB)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
