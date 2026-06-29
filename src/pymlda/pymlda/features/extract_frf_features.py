from pymlda.features.frf_features import (
    frf_mean,
    frf_std,
    frf_energy,
    peak_frequency,
    peak_magnitude,
    number_of_peaks,
    modal_frequencies,
)

import numpy as np


def extract_frf_features(
    frequency,
    magnitude,
    phase=None,
    n_modes=3
):
    """
    Extract features from Frequency Response Functions (FRFs).

    Parameters
    ----------
    frequency : array-like
        Frequency vector

    magnitude : array-like
        FRF magnitude

    phase : array-like, optional
        FRF phase

    n_modes : int
        Number of modal frequencies to extract

    Returns
    -------
    dict
        Dictionary containing FRF features
    """

    features = {}

    # =========================
    # Global FRF statistics
    # =========================

    features["frf_mean"] = frf_mean(magnitude)
    features["frf_std"] = frf_std(magnitude)
    features["frf_energy"] = frf_energy(magnitude)

    # =========================
    # Main resonance
    # =========================

    features["peak_frequency"] = peak_frequency(
        frequency,
        magnitude
    )

    features["peak_magnitude"] = peak_magnitude(
        magnitude
    )

    # =========================
    # Resonance count
    # =========================

    features["number_of_peaks"] = number_of_peaks(
        frequency,
        magnitude
    )

    # =========================
    # Modal frequencies
    # =========================

    modal_freqs, modal_mags = modal_frequencies(
        frequency,
        magnitude,
        n_modes=n_modes
    )

    for i in range(n_modes):

        if i < len(modal_freqs):

            features[f"mode_{i+1}_freq"] = modal_freqs[i]
            features[f"mode_{i+1}_mag"] = modal_mags[i]

        else:

            features[f"mode_{i+1}_freq"] = np.nan
            features[f"mode_{i+1}_mag"] = np.nan

    # =========================
    # Phase statistics
    # =========================

    if phase is not None:

        features["phase_mean"] = np.mean(phase)
        features["phase_std"] = np.std(phase)

    return features