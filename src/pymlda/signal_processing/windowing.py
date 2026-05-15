import numpy as np


def sliding_window(signal, window_size, overlap=0.5):
    """
    Generate sliding windows from a signal.

    Parameters
    ----------
    signal : array-like
        Input signal

    window_size : int
        Number of samples per window

    overlap : float
        Overlap ratio between 0 and 1

    Returns
    -------
    windows : list
        List of signal windows
    """

    signal = np.asarray(signal)

    step = int(window_size * (1 - overlap))

    if step <= 0:
        raise ValueError("Overlap too large")

    windows = []

    for start in range(0, len(signal) - window_size + 1, step):

        end = start + window_size

        windows.append(signal[start:end])

    return windows