"""
pymlda.features

Feature extraction module for machine learning and signal analysis.
"""

# ==========================================================
# High-level API
# ==========================================================

from .extract_time_features import extract_time_features
from .extract_spectral_features import extract_spectral_features
from .extract_frf_features import extract_frf_features

from .windowed_extractor import extract_features_windowed

# ==========================================================
# Low-level features
# ==========================================================

from .time_features import *
from .spectral_features import *
from .frf_features import *

# ==========================================================
# Public API
# ==========================================================

__all__ = [

    # Extractors
    "extract_time_features",
    "extract_spectral_features",
    "extract_frf_features",
    "extract_features_windowed",

]

# from .time_features import (
   #  rms,
   #  mean,
    # median,
  #   std,
  #   variance,
  #   maximum,
 #    minimum,
  #   amplitude,
  #   peak,
  #   peak_to_peak,
  #   mean_absolute_value,
  #   crest_factor,
  #   shape_factor,
  #   impulse_factor,
  #   clearance_factor,
   #  energy,
   #  centered_energy,
   #  skewness,
   #  kurtosis,
  #  entropy,
  # #   moment,
  #   signal_rate,
  #   zero_crossing_rate,
# )

# from .spectral_features import (
   #  spectral_mean,
    # spectral_variance,
 #    spectral_std,
  #   spectral_energy,
  #   spectral_entropy,
  #   spectral_skewness,
  #   spectral_kurtosis,
  #   dominant_frequency,
  #   spectral_centroid,
   #  spectral_bandwidth,
  #   low_band_ratio,
   #  mid_band_ratio,
   #  high_band_ratio,
  #   spectral_rolloff,
# )

# ==========================================================
# High-level API (extractors)
# ==========================================================

# from .extractor import (
   #  extract_features,
    # extract_time_features,
   #  extract_spectral_features,
   #  extract_frf_features,
# )

# from .windowed_extractor import (
   #  extract_features_windowed,
# )