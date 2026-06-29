from .extract_time_features import extract_time_features
from .extract_spectral_features import extract_spectral_features


def extract_features(signal, fs=None, domain="all"):

    features = {}

    if domain in ["time", "all"]:
        features.update(extract_time_features(signal))

    if domain in ["spectral", "all"]:

        if fs is None:
            raise ValueError("fs must be provided for spectral features")

        features.update(extract_spectral_features(signal, fs))

    return features