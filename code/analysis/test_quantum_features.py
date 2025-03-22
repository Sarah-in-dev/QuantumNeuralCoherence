import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)

# Now try importing
from src.features.quantum_features import (
    extract_multiscale_entropy,
    compute_nonlinear_scaling,
    phase_synchronization_analysis,
    wavelet_quantum_decomposition,
    nonlinear_transfer_entropy
)

# Generate test signals
fs = 250  # Sampling frequency (Hz)
t = np.arange(0, 10, 1/fs)  # 10 seconds
f1, f2 = 10, 15  # Frequencies (Hz)
signal1 = np.sin(2 * np.pi * f1 * t) + 0.5 * np.random.randn(len(t))
signal2 = np.sin(2 * np.pi * f2 * t) + 0.5 * np.random.randn(len(t))

# Test each function
print("Testing multiscale entropy...")
mse = extract_multiscale_entropy(signal1, scale_range=5)
print(f"MSE results: {mse}")

print("\nTesting nonlinear scaling...")
hq, tau = compute_nonlinear_scaling(signal1)
print(f"Hurst exponents: {hq}")
print(f"Tau values: {tau}")

print("\nTesting phase synchronization...")
psi, entropy = phase_synchronization_analysis(signal1, signal2, fs, (5, 20))
print(f"Phase sync index: {psi}, Entropy: {entropy}")

print("\nTesting wavelet decomposition...")
coeffs, freqs, wavelet_entropy = wavelet_quantum_decomposition(signal1, fs)
print(f"Wavelet frequencies: {freqs[:5]}...")
print(f"Wavelet entropy: {wavelet_entropy[:5]}...")

print("\nTesting transfer entropy...")
te, sig = nonlinear_transfer_entropy(signal1, signal2)
print(f"Transfer entropy: {te}, Significance: {sig}")

print("\nAll tests completed!")
