import pandas as pd

from pymlda.signal_processing import sliding_window

from extract_time_features import extract_features


def extract_features_windowed(
    signal,
    fs,
    window_size,
    overlap=0.5,
    domain="all"
):
    """
    Extract features from sliding windows.

    Parameters
    ----------
    signal : array-like
        Input signal

    fs : float
        Sampling frequency

    window_size : int
        Number of samples per window

    overlap : float
        Window overlap ratio

    domain : str
        'time', 'frequency', or 'all'

    Returns
    -------
    df : pandas.DataFrame
        Feature matrix
    """

    windows = sliding_window(
        signal,
        window_size=window_size,
        overlap=overlap
    )

    feature_list = []

    for i, w in enumerate(windows):

        features = extract_features(
            w,
            fs=fs,
            domain=domain
        )

        features["window"] = i

        feature_list.append(features)

    df = pd.DataFrame(feature_list)

    return df