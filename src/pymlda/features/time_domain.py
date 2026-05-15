import numpy as np
from scipy import stats

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

def variance(signal):
    """
    Variance
    """
    signal = np.asarray(signal)
    return np.var(signal)


def maximum(signal):
    """
    Maximum value
    """
    signal = np.asarray(signal)
    return np.max(signal)


def minimum(signal):
    """
    Minimum value
    """
    signal = np.asarray(signal)
    return np.min(signal)


def amplitude(signal):
    """
    Signal amplitude
    """
    signal = np.asarray(signal)
    return (np.max(signal)-np.min(signal))/2


def peak(signal):
    """
    Peak value
    """
    signal = np.asarray(signal)
    return np.max(np.abs(signal))


def crest_factor(signal):
    """
    Crest Factor = peak/RMS
    """
    return peak(signal)/rms(signal)


def energy(signal):
    """
    Signal energy
    """
    signal = np.asarray(signal)
    return np.sum(signal**2)

def centered_energy(signal):
    """
    Energy around the mean
    """
    signal = np.asarray(signal)
    mu = np.mean(signal)
    return np.sum((signal - mu)**2)


def skewness(signal):
    """
    Signal skewness
    """
    signal = np.asarray(signal)
    return stats.skew(signal, nan_policy='omit')


def kurtosis(signal):
    """
    Kurtosis (measure of impulsiveness)
    """
    signal = np.asarray(signal)
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    return np.mean(((signal - mean_val)/std_val)**4)

def entropy(signal):
    """
    Shannon entropy
    """
    signal = np.asarray(signal)
    signal = np.abs(signal)
    signal = signal/np.sum(signal)

    return stats.entropy(signal)


def moment(signal, order=6):
    """
    Statistical moment of arbitrary order
    """
    signal = np.asarray(signal)
    return stats.moment(signal, moment=order)


def signal_rate(signal):
    """
    Signal rate of change
    """
    signal = np.asarray(signal)
    return np.mean(np.abs(np.diff(signal)))