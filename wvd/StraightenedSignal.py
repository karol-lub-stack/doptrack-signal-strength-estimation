import numpy as np
from numpy import load
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from WaterfallCurveData import update_waterfall_curve


iq_data = np.fromfile(r"C:\Users\novac\OneDrive\Desktop\Y2 Books\Q3\Project Q3\Data\FUNcube-1_39444_202601010247.fc32", dtype=np.complex64) # Data from L0 Products, which has the IQ data, already downsampled to 25 kHz
waterfall_data = np.loadtxt('C:\\Users\\novac\\Downloads\\FUNcube-1_39444_202601010247.dat', skiprows=1) # Data from L1B Products, which has the waterfall curve extracted
#waterfall_data = update_waterfall_curve(waterfall_data)
duration = 660  # seconds, total duration of the signal
sampling_rate = 25000  # Hz, sampling rate of the IQ data
dt = 1 / sampling_rate  # sampling interval

def straighten_signal(iq_data, duration, waterfall_data):
    freq, fitted_freq, time = update_waterfall_curve(waterfall_data)
    # Using data from L1B Products, which has the waterfall curve extracted
    # Interpolate frequency data
    interpolator = interp1d(time, freq, kind='linear', fill_value="extrapolate")  # linear interpolation, extrapolate outside bounds
    # Interpolated frequency values at the time points of the IQ data
    n = len(iq_data)
    # Create time vector for the IQ data based on its length and sampling rate
    t = np.arange(0, duration, dt)
    freq_interpolated = interpolator(t)  # get interpolated frequency for each sample index
    # Calculate the frequency shift needed to move the signal to 0 Hz
    delta_freq = 0 - freq_interpolated  # frequency shift needed to move the signal to 0 Hz, as the signal is already downconverted
    # Create shifted signal
    x = np.zeros(n, dtype=np.complex64)
    # To shift the signal in the time domain, we can multiply by the complex exponential
    x = iq_data * np.exp(-2j * np.pi * delta_freq * t)
    return x

#x = straighten_signal(iq_data, duration, waterfall_data)
#t = np.arange(0, duration, dt)
#plt.plot(t, np.real(x))
#plt.show()
