from pymlda.features.time_features import *


def extract_time_features(signal):

    return {
        "rms": rms(signal),
        "mean": mean(signal),
        "median": median(signal),
        "std": std(signal),
        "variance": variance(signal),

        "maximum": maximum(signal),
        "minimum": minimum(signal),

        "peak": peak(signal),
        "peak_to_peak": peak_to_peak(signal),
        "amplitude": amplitude(signal),

        "mean_absolute_value": mean_absolute_value(signal),

        "energy": energy(signal),
        "centered_energy": centered_energy(signal),

        "skewness": skewness(signal),
        "kurtosis": kurtosis(signal),
        "entropy": entropy(signal),

        "crest_factor": crest_factor(signal),
        "shape_factor": shape_factor(signal),
        "impulse_factor": impulse_factor(signal),
        "clearance_factor": clearance_factor(signal),

        "signal_rate": signal_rate(signal),
        "zero_crossing_rate": zero_crossing_rate(signal),
    }