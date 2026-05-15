from pymlda.features.time_domain import (
    rms,
    mean,
    std,
    variance,
    skewness,
    kurtosis,
    crest_factor,
    entropy,
    energy,
)

from pymlda.features.frequency_domain import (
    spectral_mean,
    spectral_variance,
    spectral_std,
    spectral_energy,
    spectral_entropy,
    dominant_frequency,
    spectral_centroid,
    spectral_rolloff,
)


def extract_features(signal, fs=None, domain="all"):
    """
    Automatic feature extraction.

    Parameters
    ----------
    signal : array-like
        Input signal

    fs : float
        Sampling frequency

    domain : str
        'time', 'frequency', or 'all'

    Returns
    -------
    features : dict
        Dictionary containing extracted features
    """

    features = {}

    # =========================
    # TIME DOMAIN
    # =========================

    if domain in ["time", "all"]:

        features["rms"] = rms(signal)
        features["mean"] = mean(signal)
        features["std"] = std(signal)
        features["variance"] = variance(signal)
        features["skewness"] = skewness(signal)
        features["kurtosis"] = kurtosis(signal)
        features["crest_factor"] = crest_factor(signal)
        features["entropy"] = entropy(signal)
        features["energy"] = energy(signal)

    # =========================
    # FREQUENCY DOMAIN
    # =========================

    if domain in ["frequency", "all"]:

        if fs is None:
            raise ValueError(
                "Sampling frequency fs must be provided "
                "for frequency-domain features."
            )

        features["spectral_mean"] = spectral_mean(signal, fs)
        features["spectral_variance"] = spectral_variance(signal, fs)
        features["spectral_std"] = spectral_std(signal, fs)
        features["spectral_energy"] = spectral_energy(signal, fs)
        features["spectral_entropy"] = spectral_entropy(signal, fs)
        features["dominant_frequency"] = dominant_frequency(signal, fs)
        features["spectral_centroid"] = spectral_centroid(signal, fs)
        features["spectral_rolloff"] = spectral_rolloff(signal, fs)

    return features