"""
Signal filtering and preprocessing functions.
"""

import numpy as np
from scipy import signal

def bandpass_filter(data, low_freq, high_freq, sampling_rate, order=4):
    """
    Apply a bandpass filter to the input data.
    
    Parameters:
    -----------
    data : array-like
        Input signal, shape (n_channels, n_samples) or (n_samples,)
    low_freq : float
        Lower cutoff frequency in Hz
    high_freq : float
        Upper cutoff frequency in Hz
    sampling_rate : float
        Sampling rate in Hz
    order : int, optional
        Filter order. Default is 4.
        
    Returns:
    --------
    filtered_data : array-like
        Filtered signal with same shape as input
    """
    nyq = 0.5 * sampling_rate
    low = low_freq / nyq
    high = high_freq / nyq
    
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Handle multi-channel data
    if len(data.shape) == 2:
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered_data[i, :] = signal.filtfilt(b, a, data[i, :])
    else:
        filtered_data = signal.filtfilt(b, a, data)
    
    return filtered_data

def notch_filter(data, freq, sampling_rate, q=30):
    """
    Apply a notch filter to remove specific frequency (e.g., 60Hz line noise).
    
    Parameters:
    -----------
    data : array-like
        Input signal, shape (n_channels, n_samples) or (n_samples,)
    freq : float
        Frequency to remove in Hz
    sampling_rate : float
        Sampling rate in Hz
    q : float, optional
        Quality factor. Default is 30.
        
    Returns:
    --------
    filtered_data : array-like
        Filtered signal with same shape as input
    """
    nyq = 0.5 * sampling_rate
    freq_normalized = freq / nyq
    
    b, a = signal.iirnotch(freq_normalized, q)
    
    # Handle multi-channel data
    if len(data.shape) == 2:
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered_data[i, :] = signal.filtfilt(b, a, data[i, :])
    else:
        filtered_data = signal.filtfilt(b, a, data)
    
    return filtered_data
