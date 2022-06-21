from scipy.signal import butter, filtfilt


def bandpass_filter(data, low, high, fs, order=5,axis=1):
    b, a = butter(order, [low, high], fs=fs, btype='bandpass', analog=False)
    y = filtfilt(b, a, data, axis=axis)
    return y