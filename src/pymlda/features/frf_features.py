import numpy as np
from scipy.signal import find_peaks

#Estatistical features of FRF
def frf_mean(magnitude):
    return np.mean(magnitude)

def frf_std(magnitude):
    return np.std(magnitude)

def frf_energy(magnitude):
    return np.sum(magnitude**2)


#Main peak
def peak_frequency(freq, magnitude):
    idx = np.argmax(magnitude)
    return freq[idx]

def peak_values(freq, magnitude, n_peaks=1):
    peaks, _ = find_peaks(magnitude, height=np.max(magnitude)*0.5)
    peak_freqs = freq[peaks]
    peak_mags = magnitude[peaks]

    # Sort by magnitude and return the top n_peaks
    sorted_indices = np.argsort(peak_mags)[::-1][:n_peaks]
    return peak_freqs[sorted_indices], peak_mags[sorted_indices]

#Peak magnitude
def peak_magnitude(magnitude):
    return np.max(magnitude)

#Resonance frequency number
def number_of_peaks(
    freq,
    magnitude,
    prominence=1):

    peaks, _ = find_peaks(
        magnitude,
        prominence=prominence
    )
    return len(peaks)

# Modal frequency
def modal_frequencies(
    freq,
    magnitude,
    n_modes=3
):
    peaks, _ = find_peaks(magnitude)
    peak_freqs = freq[peaks]
    peak_mags = magnitude[peaks]

    # Sort by magnitude and return the top n_modes
    sorted_indices = np.argsort(peak_mags)[::-1][:n_modes]
    return peak_freqs[sorted_indices], peak_mags[sorted_indices]


#Frequência ponderada (centroide da FRF)
def frf_centroid(freq, magnitude):

    mag = np.abs(magnitude)

    return np.sum(freq * mag) / np.sum(mag)


#Variância espectral da FRF
def frf_bandwidth(freq, magnitude):

    centroid = frf_centroid(freq, magnitude)

    mag = np.abs(magnitude)

    return np.sqrt(
        np.sum(((freq-centroid)**2)*mag)
        / np.sum(mag)
    )

def frf_max(magnitude):
    return np.max(magnitude)


def frf_min(magnitude):
    return np.min(magnitude)


def frf_range(magnitude):
    return np.max(magnitude) - np.min(magnitude)