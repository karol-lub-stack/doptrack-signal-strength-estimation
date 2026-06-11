import numpy as np
import matplotlib.pyplot as plt

duration = 660  # seconds, total duration of the signal
arr = 10 * np.log10(np.abs(np.loadtxt("Power_list_updated.csv", delimiter=",")) + 1e-20)
n = len(arr)
print(n)
time = np.linspace(0, duration, n)  # time points corresponding to the data
window_size = 500  # size of the moving average window


def moving_average(arr, window_size):
    """Calculate the moving average of the given data with the specified window size."""
    #return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    print(len(arr))
    i = 0
    moving_averages = []
    while i < len(arr) - window_size + 1:
        # Store elements from i to i+window_size in list to get the current window
        window = arr[i : i + window_size]
        # Calculate the average of current window
        window_average = round(sum(window) / window_size, 2)
        # Store the average of current window in moving average list
        moving_averages.append(window_average)
        # Shift window to right by one position
        i += 1
        # Print progress every 100 iterations
        if i % 100 == 0:  # Print progress every 100 iterations
            print(f"Progress: {i/(len(arr) - window_size + 1) * 100:.2f}%")
    return moving_averages

moving_averages = moving_average(arr, window_size)
#print(moving_averages)
plt.plot(time, arr, label='Original Data')
plt.plot(time[window_size-1:], moving_averages, label='Moving Average')
plt.show()
