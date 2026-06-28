import pandas as pd

from pymlda.signal_processing import sliding_window
from .extractor import extract_features


def extract_features_windowed(
    signal,
    fs=None,
    window_size=1024,
    overlap=0.5,
    domain="all"
):

    windows = sliding_window(
        signal,
        window_size=window_size,
        overlap=overlap
    )

    feature_list = []

    for i, w in enumerate(windows):

        features = extract_features(
            signal=w,
            fs=fs,
            domain=domain
        )

        features["window"] = i

        feature_list.append(features)

    return pd.DataFrame(feature_list)