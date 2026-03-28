import numpy as np

def rms(signal):
    """
    Root Mean Square (RMS)
    """
    signal = np.asarray(signal)
    return np.sqrt(np.mean(signal**2))


def mean(signal):
    """
    Mean value
    """
    signal = np.asarray(signal)
    return np.mean(signal)


def std(signal):
    """
    Standard deviation
    """
    signal = np.asarray(signal)
    return np.std(signal)


def peak(signal):
    """
    Peak value
    """
    signal = np.asarray(signal)
    return np.max(np.abs(signal))


def crest_factor(signal):
    """
    Crest Factor = peak / RMS
    """
    return peak(signal) / rms(signal)


def kurtosis(signal):
    """
    Kurtosis (measure of impulsiveness)
    """
    signal = np.asarray(signal)
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    return np.mean(((signal - mean_val) / std_val) ** 4)