"""
Quantum feature extraction functions for neural data analysis.
"""

import numpy as np
from scipy import signal, stats
import pywt
import warnings

def extract_multiscale_entropy(data, m=2, r=0.15, scale_range=10):
    """
    Calculate multiscale entropy for the input signal.
    
    Parameters:
    -----------
    data : array-like
        Input time series
    m : int
        Embedding dimension
    r : float
        Tolerance value (typically 0.1-0.25 * std of the data)
    scale_range : int
        Number of timescales to compute
        
    Returns:
    --------
    entropy_values : array
        Entropy values for each timescale
    """
    # Normalize the data
    data = np.array(data)
    data = (data - np.mean(data)) / np.std(data)
    
    # Initialize array to store entropy values
    entropy_values = np.zeros(scale_range)
    
    # Calculate sample entropy for each scale
    for scale in range(1, scale_range + 1):
        # Coarse-grain the time series
        coarse_grained = _coarse_grain(data, scale)
        
        # Calculate sample entropy
        entropy_values[scale-1] = _sample_entropy(coarse_grained, m, r)
    
    return entropy_values

def _coarse_grain(data, scale):
    """
    Perform coarse-graining for multiscale entropy.
    
    Parameters:
    -----------
    data : array-like
        Input time series
    scale : int
        Scale factor
        
    Returns:
    --------
    coarse_grained : array
        Coarse-grained time series
    """
    n = len(data)
    coarse_n = int(np.floor(n / scale))
    coarse_grained = np.zeros(coarse_n)
    
    for i in range(coarse_n):
        coarse_grained[i] = np.mean(data[i*scale:(i+1)*scale])
    
    return coarse_grained

def _sample_entropy(data, m, r):
    """
    Calculate sample entropy of a time series.
    
    Parameters:
    -----------
    data : array-like
        Input time series
    m : int
        Embedding dimension
    r : float
        Tolerance value
        
    Returns:
    --------
    sample_entropy : float
        Sample entropy value
    """
    n = len(data)
    if n < 2 * m + 1:
        return np.nan
    
    # Form m and m+1 dimensional patterns
    patterns_m = np.zeros((n-m+1, m))
    patterns_m1 = np.zeros((n-m, m+1))
    
    for i in range(n-m+1):
        patterns_m[i] = data[i:i+m]
        if i < n-m:
            patterns_m1[i] = data[i:i+m+1]
    
    # Count similar patterns
    count_m = 0
    count_m1 = 0
    
    for i in range(n-m):
        # For m-dimensional vectors
        temp_m = np.abs(patterns_m - patterns_m[i])
        temp_m = np.max(temp_m, axis=1)
        count_m += np.sum(temp_m < r) - 1  # exclude self-match
        
        # For (m+1)-dimensional vectors
        temp_m1 = np.abs(patterns_m1 - patterns_m1[i])
        temp_m1 = np.max(temp_m1, axis=1)
        count_m1 += np.sum(temp_m1 < r) - 1  # exclude self-match
    
    # Calculate sample entropy
    if count_m == 0 or count_m1 == 0:
        return np.nan
    
    return -np.log(count_m1 / count_m)

def compute_nonlinear_scaling(data, q_values=None, min_scale=4, max_scale=64, n_scales=10):
    """
    Compute nonlinear scaling exponents (multifractal analysis).
    
    Parameters:
    -----------
    data : array-like
        Input time series
    q_values : array-like or None
        Moment orders to compute
    min_scale : int
        Minimum scale for analysis
    max_scale : int
        Maximum scale for analysis
    n_scales : int
        Number of scales to compute
        
    Returns:
    --------
    hq : array
        Generalized Hurst exponents for each q
    tau : array
        Scaling function tau(q)
    """
    if q_values is None:
        q_values = np.array([-5, -3, -1, 0.01, 1, 3, 5])
    
    # Create scales logarithmically spaced
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales).astype(int)
    
    # Initialize arrays
    fluctuations = np.zeros((len(scales), len(q_values)))
    
    # Normalize the data
    data = (data - np.mean(data)) / np.std(data)
    
    # Calculate the fluctuation function
    for i, scale in enumerate(scales):
        # Get profiles for overlapping windows
        profiles = _get_profiles(data, scale)
        
        # Calculate fluctuations for each q
        for j, q in enumerate(q_values):
            if abs(q) < 0.05:  # q ≈ 0
                fluctuations[i, j] = np.exp(0.5 * np.mean(np.log(profiles**2)))
            else:
                fluctuations[i, j] = np.mean(profiles**q)**(1/q)
    
    # Calculate Hurst exponents and tau
    hq = np.zeros(len(q_values))
    tau = np.zeros(len(q_values))
    
    for j in range(len(q_values)):
        # Linear fit on log-log scale
        log_scales = np.log(scales)
        log_fluct = np.log(fluctuations[:, j])
        
        # Calculate slope (Hurst exponent)
        hq[j], _, _, _, _ = stats.linregress(log_scales, log_fluct)
        
        # Calculate tau(q)
        tau[j] = q_values[j] * hq[j] - 1
    
    return hq, tau

def _get_profiles(data, scale):
    """
    Get profiles of data for multifractal analysis.
    
    Parameters:
    -----------
    data : array-like
        Input time series
    scale : int
        Scale size
        
    Returns:
    --------
    fluctuations : array
        Fluctuation magnitudes
    """
    n = len(data)
    n_windows = n // scale
    
    if n_windows == 0:
        return np.array([0])
    
    # Compute cumulative sum
    y = np.cumsum(data - np.mean(data))
    
    # Initialize fluctuations
    fluctuations = np.zeros(n_windows)
    
    # Calculate fluctuations for each window
    for i in range(n_windows):
        start = i * scale
        end = (i + 1) * scale
        
        # Extract window
        y_window = y[start:end]
        
        # Fit linear trend
        x = np.arange(scale)
        p = np.polyfit(x, y_window, 1)
        trend = np.polyval(p, x)
        
        # Calculate fluctuation as RMS of detrended window
        fluctuations[i] = np.sqrt(np.mean((y_window - trend)**2))
    
    return fluctuations

def phase_synchronization_analysis(signal1, signal2, fs, f_band, visualize=False):
    """
    Calculate phase synchronization index and phase difference entropy.
    
    Parameters:
    -----------
    signal1 : array-like
        First input signal
    signal2 : array-like
        Second input signal
    fs : float
        Sampling frequency in Hz
    f_band : tuple
        Frequency band of interest (low, high) in Hz
    visualize : bool
        Whether to generate visualizations
        
    Returns:
    --------
    psi : float
        Phase synchronization index (0-1)
    entropy : float
        Phase difference entropy
    """
    # Filter the signals in the frequency band
    from scipy.signal import butter, filtfilt
    
    # Normalize frequencies
    nyq = 0.5 * fs
    low = f_band[0] / nyq
    high = f_band[1] / nyq
    
    # Design filter
    b, a = butter(3, [low, high], btype='band')
    
    # Apply filter
    signal1_filt = filtfilt(b, a, signal1)
    signal2_filt = filtfilt(b, a, signal2)
    
    # Compute analytic signal using Hilbert transform
    signal1_hilbert = signal.hilbert(signal1_filt)
    signal2_hilbert = signal.hilbert(signal2_filt)
    
    # Extract instantaneous phases
    phase1 = np.angle(signal1_hilbert)
    phase2 = np.angle(signal2_hilbert)
    
    # Compute phase difference
    phase_diff = phase1 - phase2
    
    # Wrap phase difference to [-pi, pi]
    phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
    
    # Compute phase synchronization index (mean resultant length)
    complex_phase_diff = np.exp(1j * phase_diff)
    psi = np.abs(np.mean(complex_phase_diff))
    
    # Compute phase difference entropy
    n_bins = 24  # Number of bins for the histogram
    hist, _ = np.histogram(phase_diff, bins=n_bins, range=(-np.pi, np.pi), density=True)
    
    # Calculate Shannon entropy
    hist = hist[hist > 0]  # Only consider non-zero bins
    entropy = -np.sum(hist * np.log(hist))
    
    return psi, entropy

def wavelet_quantum_decomposition(signal, fs, wavelet='cmor1.5-1.0', n_freqs=64, visualize=False):
    """
    Perform wavelet-based quantum decomposition of a signal.
    
    Parameters:
    -----------
    signal : array-like
        Input signal
    fs : float
        Sampling frequency in Hz
    wavelet : str
        Wavelet to use for the transform
    n_freqs : int
        Number of frequency bins
    visualize : bool
        Whether to generate visualizations
        
    Returns:
    --------
    cwt_coef : array
        Complex wavelet transform coefficients
    freqs : array
        Frequencies corresponding to the coefficients
    entropy : array
        Wavelet entropy at each scale
    """
    # Define frequencies for wavelet transform (logarithmically spaced)
    max_freq = fs / 2  # Nyquist frequency
    min_freq = 1.0     # Minimum frequency
    freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), n_freqs)
    
    # Normalize signal
    signal = (signal - np.mean(signal)) / np.std(signal)
    
    # Compute continuous wavelet transform
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cwt_coef, _ = pywt.cwt(signal, freqs, wavelet, 1/fs)
    
    # Calculate wavelet entropy (measure of signal complexity)
    # Entropy based on energy distribution across scales
    energy = np.abs(cwt_coef)**2
    total_energy = np.sum(energy, axis=1)
    
    # Avoid division by zero
    total_energy = np.where(total_energy == 0, 1e-10, total_energy)
    
    # Normalize to get probability distribution
    p = energy / total_energy[:, np.newaxis]
    
    # Calculate entropy for each scale
    entropy = np.zeros(len(freqs))
    for i in range(len(freqs)):
        # Only consider non-zero probabilities
        p_nonzero = p[i, p[i] > 0]
        entropy[i] = -np.sum(p_nonzero * np.log2(p_nonzero))
    
    return cwt_coef, freqs, entropy

def nonlinear_transfer_entropy(source, target, lag=1, k=1, n_bins=10):
    """
    Calculate nonlinear transfer entropy from source to target signal.
    
    Parameters:
    -----------
    source : array-like
        Source signal
    target : array-like
        Target signal
    lag : int
        Time lag to compute TE
    k : int
        History length
    n_bins : int
        Number of bins for discretization
        
    Returns:
    --------
    te : float
        Transfer entropy in bits
    significance : float
        Statistical significance of TE
    """
    # Normalize signals
    source = (source - np.mean(source)) / np.std(source)
    target = (target - np.mean(target)) / np.std(target)
    
    # Discretize the signals
    source_disc = _discretize(source, n_bins)
    target_disc = _discretize(target, n_bins)
    
    # Prepare matrices for joint probabilities
    n = len(source_disc) - lag - k
    
    if n <= 0:
        return 0, 0
    
    # Prepare state vectors
    target_future = target_disc[lag+k:]
    target_past = np.zeros((n, k))
    source_past = source_disc[lag:lag+n]
    
    for i in range(k):
        target_past[:, i] = target_disc[i:i+n]
    
    # Calculate probabilities
    p_tar_fut_tar_past = np.zeros((n_bins, n_bins, n_bins))
    p_tar_past_src_past = np.zeros((n_bins, n_bins))
    p_tar_fut_tar_past_src_past = np.zeros((n_bins, n_bins, n_bins))
    p_tar_past = np.zeros(n_bins)
    
    # Count occurrences
    for i in range(n):
        if k == 1:
            t_past = int(target_past[i, 0])
        else:
            # For higher k, we need a better encoding strategy
            t_past = int(np.mean(target_past[i, :]))
        
        t_fut = int(target_future[i])
        s_past = int(source_past[i])
        
        p_tar_fut_tar_past[t_fut, t_past, s_past] += 1
        p_tar_past_src_past[t_past, s_past] += 1
        p_tar_fut_tar_past[t_fut, t_past, :] += 1
        p_tar_past[t_past] += 1
    
    # Normalize counts to probabilities
    p_tar_fut_tar_past /= n
    p_tar_past_src_past /= n
    p_tar_fut_tar_past /= n
    p_tar_past /= n
    
    # Calculate transfer entropy
    te = 0
    
    for i in range(n_bins):
        for j in range(n_bins):
            for k in range(n_bins):
                if (p_tar_fut_tar_past[i, j, k] > 0 and 
                    p_tar_past_src_past[j, k] > 0 and 
                    p_tar_fut_tar_past[i, j, 0] > 0 and
                    p_tar_past[j] > 0):
                    
                    te += p_tar_fut_tar_past[i, j, k] * np.log2(
                        p_tar_fut_tar_past[i, j, k] * p_tar_past[j] / 
                        (p_tar_past_src_past[j, k] * p_tar_fut_tar_past[i, j, 0])
                    )
    
    # Calculate significance (using surrogate data)
    n_surrogates = 10
    surrogate_te = np.zeros(n_surrogates)
    
    for s in range(n_surrogates):
        # Create phase-randomized surrogate
        surrogate_source = _create_surrogate(source)
        surrogate_source_disc = _discretize(surrogate_source, n_bins)
        surrogate_source_past = surrogate_source_disc[lag:lag+n]
        
        # Calculate TE for surrogate
        p_tar_fut_tar_past_surr = np.zeros((n_bins, n_bins, n_bins))
        p_tar_past_src_past_surr = np.zeros((n_bins, n_bins))
        
        for i in range(n):
            if k == 1:
                t_past = int(target_past[i, 0])
            else:
                t_past = int(np.mean(target_past[i, :]))
            
            t_fut = int(target_future[i])
            s_past_surr = int(surrogate_source_past[i])
            
            p_tar_fut_tar_past_surr[t_fut, t_past, s_past_surr] += 1
            p_tar_past_src_past_surr[t_past, s_past_surr] += 1
        
        p_tar_fut_tar_past_surr /= n
        p_tar_past_src_past_surr /= n
        
        # Calculate TE
        surrogate_te[s] = 0
        for i in range(n_bins):
            for j in range(n_bins):
                for l in range(n_bins):
                    if (p_tar_fut_tar_past_surr[i, j, l] > 0 and 
                        p_tar_past_src_past_surr[j, l] > 0 and 
                        p_tar_fut_tar_past[i, j, 0] > 0 and
                        p_tar_past[j] > 0):
                        
                        surrogate_te[s] += p_tar_fut_tar_past_surr[i, j, l] * np.log2(
                            p_tar_fut_tar_past_surr[i, j, l] * p_tar_past[j] / 
                            (p_tar_past_src_past_surr[j, l] * p_tar_fut_tar_past[i, j, 0])
                        )
    
    # Calculate Z-score
    if np.std(surrogate_te) > 0:
        significance = (te - np.mean(surrogate_te)) / np.std(surrogate_te)
    else:
        significance = 0
    
    return te, significance

def _discretize(signal, n_bins):
    """
    Discretize a continuous signal using uniform binning.
    
    Parameters:
    -----------
    signal : array-like
        Input signal
    n_bins : int
        Number of bins
        
    Returns:
    --------
    discretized : array
        Discretized signal with values 0 to n_bins-1
    """
    # Ensure values between 0 and n_bins-1
    min_val = np.min(signal)
    max_val = np.max(signal)
    
    if min_val == max_val:
        return np.zeros_like(signal, dtype=int)
    
    # Scale to [0, n_bins-1]
    scaled = (signal - min_val) / (max_val - min_val) * (n_bins - 1)
    discretized = np.floor(scaled).astype(int)
    
    # Ensure values are within range
    discretized[discretized < 0] = 0
    discretized[discretized >= n_bins] = n_bins - 1
    
    return discretized

def _create_surrogate(signal):
    """
    Create phase-randomized surrogate data.
    
    Parameters:
    -----------
    signal : array-like
        Input signal
        
    Returns:
    --------
    surrogate : array
        Phase-randomized surrogate
    """
    n = len(signal)
    
    # Compute FFT
    fft_signal = np.fft.fft(signal)
    
    # Randomize phases but keep amplitudes
    magnitudes = np.abs(fft_signal)
    phases = np.angle(fft_signal)
    
    # Generate random phases but keep DC component and Nyquist frequency unchanged
    random_phases = np.random.uniform(0, 2*np.pi, n)
    random_phases[0] = phases[0]  # Keep DC phase
    if n % 2 == 0:
        random_phases[n//2] = phases[n//2]  # Keep Nyquist phase
    
    # Create symmetrical phases for real output
    random_phases[1:n//2] = random_phases[1:n//2]
    random_phases[n//2+1:] = -random_phases[1:n//2][::-1]
    
    # Combine with original magnitudes
    fft_surrogate = magnitudes * np.exp(1j * random_phases)
    
    # Inverse FFT
    surrogate = np.real(np.fft.ifft(fft_surrogate))
    
    return surrogate
