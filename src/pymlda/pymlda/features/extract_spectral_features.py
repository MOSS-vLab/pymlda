from pymlda.features.spectral_features import *


def extract_spectral_features(
    signal,
    fs
):
    """
    Extract spectral features from a signal.
    """

    f, Pxx = power_spectral_density(
        signal,
        fs
    )

    return {

        # Statistics
        "spectral_mean":
            spectral_mean(Pxx),

        "spectral_variance":
            spectral_variance(Pxx),

        "spectral_std":
            spectral_std(Pxx),

        "spectral_energy":
            spectral_energy(Pxx),

        "spectral_entropy":
            spectral_entropy(Pxx),

        "spectral_skewness":
            spectral_skewness(Pxx),

        "spectral_kurtosis":
            spectral_kurtosis(Pxx),

        # Frequency location
        "dominant_frequency":
            dominant_frequency(
                f,
                Pxx
            ),

        "spectral_centroid":
            spectral_centroid(
                f,
                Pxx
            ),

        "spectral_bandwidth":
            spectral_bandwidth(
                f,
                Pxx
            ),

        "spectral_rolloff":
            spectral_rolloff(
                f,
                Pxx
            ),

        # Band ratios
        "low_band_ratio":
            low_band_ratio(
                f,
                Pxx
            ),

        "mid_band_ratio":
            mid_band_ratio(
                f,
                Pxx
            ),

        "high_band_ratio":
            high_band_ratio(
                f,
                Pxx
            ),
    }