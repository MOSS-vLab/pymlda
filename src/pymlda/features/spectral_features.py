import numpy as np
from scipy import signal
from scipy import stats


def power_spectral_density(x, fs):
    """
    Compute Welch Power Spectral Density.
    """

    x = np.asarray(x)

    f, Pxx = signal.welch(
        x,
        fs=fs,
        nperseg=min(1024, max(8, len(x)//2))
    )

    Pxx = Pxx + 1e-12

    return f, Pxx


# ==========================================================
# Spectral statistics
# ==========================================================

def spectral_mean(Pxx):
    return np.mean(Pxx)


def spectral_variance(Pxx):
    return np.var(Pxx)


def spectral_std(Pxx):
    return np.std(Pxx)


def spectral_energy(Pxx):
    return np.sum(Pxx)


def spectral_entropy(Pxx):

    p = Pxx / np.sum(Pxx)

    return -np.sum(
        p * np.log2(p + 1e-12)
    )


def spectral_skewness(Pxx):

    return stats.skew(
        Pxx,
        bias=False
    )


def spectral_kurtosis(Pxx):

    return stats.kurtosis(
        Pxx,
        fisher=False,
        bias=False
    )


# ==========================================================
# Spectral location features
# ==========================================================

def dominant_frequency(f, Pxx):

    return f[np.argmax(Pxx)]


def spectral_centroid(f, Pxx):

    return np.sum(f * Pxx) / (
        np.sum(Pxx) + 1e-12
    )


def spectral_bandwidth(f, Pxx):

    centroid = spectral_centroid(
        f,
        Pxx
    )

    return np.sqrt(
        np.sum(
            ((f-centroid)**2)*Pxx
        )
        /
        (np.sum(Pxx)+1e-12)
    )


def spectral_rolloff(
    f,
    Pxx,
    roll_percent=0.85
):

    cumulative = np.cumsum(Pxx)

    threshold = (
        roll_percent *
        cumulative[-1]
    )

    return f[
        np.searchsorted(
            cumulative,
            threshold
        )
    ]


# ==========================================================
# Band-energy ratios
# ==========================================================

def band_energy_ratio(
    f,
    Pxx,
    band
):

    low, high = band

    idx = (
        (f >= low) &
        (f < high)
    )

    band_energy = np.sum(
        Pxx[idx]
    )

    total_energy = np.sum(Pxx)

    return band_energy / (
        total_energy + 1e-12
    )


def low_band_ratio(f, Pxx):

    return band_energy_ratio(
        f,
        Pxx,
        (0.5, 5)
    )


def mid_band_ratio(f, Pxx):

    return band_energy_ratio(
        f,
        Pxx,
        (5, 20)
    )


def high_band_ratio(f, Pxx):

    return band_energy_ratio(
        f,
        Pxx,
        (20, np.inf)
    )