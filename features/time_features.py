import numpy as np
from scipy import stats


# ==========================================================
# Basic statistics
# ==========================================================

def mean(signal):
    """Mean value"""
    signal = np.asarray(signal)
    return np.mean(signal)


def median(signal):
    """Median value"""
    signal = np.asarray(signal)
    return np.median(signal)


def std(signal):
    """Standard deviation"""
    signal = np.asarray(signal)
    return np.std(signal)


def variance(signal):
    """Variance"""
    signal = np.asarray(signal)
    return np.var(signal)


def maximum(signal):
    """Maximum value"""
    signal = np.asarray(signal)
    return np.max(signal)


def minimum(signal):
    """Minimum value"""
    signal = np.asarray(signal)
    return np.min(signal)


# ==========================================================
# Amplitude-related features
# ==========================================================

def peak(signal):
    """Peak value"""
    signal = np.asarray(signal)
    return np.max(np.abs(signal))


def peak_to_peak(signal):
    """Peak-to-peak value"""
    signal = np.asarray(signal)
    return np.max(signal) - np.min(signal)


def amplitude(signal):
    """Signal amplitude"""
    signal = np.asarray(signal)
    return peak_to_peak(signal) / 2


def rms(signal):
    """Root Mean Square (RMS)"""
    signal = np.asarray(signal)
    return np.sqrt(np.mean(signal**2))


def mean_absolute_value(signal):
    """Mean Absolute Value (MAV)"""
    signal = np.asarray(signal)
    return np.mean(np.abs(signal))


# ==========================================================
# Energy features
# ==========================================================

def energy(signal):
    """Signal energy"""
    signal = np.asarray(signal)
    return np.sum(signal**2)


def centered_energy(signal):
    """Energy around the mean"""
    signal = np.asarray(signal)
    mu = np.mean(signal)
    return np.sum((signal - mu)**2)


# ==========================================================
# Statistical shape
# ==========================================================

def skewness(signal):
    """Signal skewness"""
    signal = np.asarray(signal)
    return stats.skew(signal, nan_policy="omit")


def kurtosis(signal):
    """Signal kurtosis"""
    signal = np.asarray(signal)

    return stats.kurtosis(
        signal,
        fisher=False,
        bias=False
    )


def entropy(signal):
    """Shannon entropy"""
    signal = np.asarray(signal)

    signal = np.abs(signal)
    signal = signal / (np.sum(signal) + 1e-12)

    return stats.entropy(signal)


def moment(signal, order=6):
    """Statistical moment of arbitrary order"""
    signal = np.asarray(signal)
    return stats.moment(signal, moment=order)


# ==========================================================
# Diagnostic indicators
# ==========================================================

def crest_factor(signal):
    """Crest Factor = Peak / RMS"""
    return peak(signal) / (rms(signal) + 1e-12)


def shape_factor(signal):
    """Shape Factor = RMS / MAV"""
    return rms(signal) / (
        mean_absolute_value(signal) + 1e-12
    )


def impulse_factor(signal):
    """Impulse Factor = Peak / MAV"""
    return peak(signal) / (
        mean_absolute_value(signal) + 1e-12
    )


def clearance_factor(signal):
    """
    Clearance Factor
    """
    signal = np.asarray(signal)

    denominator = (
        np.mean(np.sqrt(np.abs(signal))) ** 2
    )

    return peak(signal) / (
        denominator + 1e-12
    )


# ==========================================================
# Dynamic features
# ==========================================================

def signal_rate(signal):
    """Mean absolute rate of change"""
    signal = np.asarray(signal)

    return np.mean(
        np.abs(np.diff(signal))
    )


def zero_crossing_rate(signal):
    """Zero Crossing Rate"""
    signal = np.asarray(signal)

    return (
        np.sum(
            np.diff(np.sign(signal)) != 0
        )
        / len(signal)
    )