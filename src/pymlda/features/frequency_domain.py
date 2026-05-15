import numpy as np
from scipy import signal


def power_spectral_density(x, fs):
    """
    Compute Welch Power Spectral Density
    """

    x = np.asarray(x)

    f, Pxx = signal.welch(
        x,
        fs=fs,
        nperseg=min(1024, max(8, len(x)//2))
    )

    Pxx = Pxx + 1e-12

    return f, Pxx


def spectral_mean(x, fs):

    _, Pxx = power_spectral_density(x, fs)

    return np.mean(Pxx)


def spectral_variance(x, fs):

    _, Pxx = power_spectral_density(x, fs)

    return np.var(Pxx)


def spectral_std(x, fs):

    _, Pxx = power_spectral_density(x, fs)

    return np.std(Pxx)


def spectral_energy(x, fs):

    _, Pxx = power_spectral_density(x, fs)

    return np.mean(Pxx**2)


def spectral_entropy(x, fs):

    _, Pxx = power_spectral_density(x, fs)

    p = Pxx / np.sum(Pxx)

    return -np.sum(p * np.log2(p + 1e-12))


def spectral_skewness(x, fs):

    _, Pxx = power_spectral_density(x, fs)

    mu = np.mean(Pxx)
    sigma = np.std(Pxx)

    return np.mean(((Pxx - mu)/(sigma + 1e-12))**3)


def spectral_kurtosis(x, fs):

    _, Pxx = power_spectral_density(x, fs)

    mu = np.mean(Pxx)
    sigma = np.std(Pxx)

    return np.mean(((Pxx - mu)/(sigma + 1e-12))**4) - 3


def dominant_frequency(x, fs):

    f, Pxx = power_spectral_density(x, fs)

    return f[np.argmax(Pxx)]


def spectral_centroid(x, fs):

    f, Pxx = power_spectral_density(x, fs)

    return np.sum(f * Pxx) / np.sum(Pxx)


def band_energy_ratio(x, fs, band):

    f, Pxx = power_spectral_density(x, fs)

    low, high = band

    idx = (f >= low) & (f < high)

    band_energy = np.sum(Pxx[idx])

    total_energy = np.sum(Pxx)

    return band_energy / (total_energy + 1e-12)


def low_band_ratio(x, fs):

    return band_energy_ratio(x, fs, (0.5, 5))


def mid_band_ratio(x, fs):

    return band_energy_ratio(x, fs, (5, 20))


def high_band_ratio(x, fs):

    return band_energy_ratio(x, fs, (20, np.inf))


def spectral_rolloff(x, fs, roll_percent=0.85):

    f, Pxx = power_spectral_density(x, fs)

    cumulative = np.cumsum(Pxx)

    threshold = roll_percent * cumulative[-1]

    return f[np.searchsorted(cumulative, threshold)]