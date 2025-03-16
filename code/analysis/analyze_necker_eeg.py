#!/usr/bin/env python
"""
Analysis script for Necker Cube EEG dataset (Maksimenko et al.)
This script processes EEG data recorded during Necker cube perception with varying ambiguity levels.

Author: Sarah Davidson
Date: March 16, 2025
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Add the source directory to path to access custom modules
sys.path.append('/blue/mcb4324/share/sdavidson2/QuantumNeuralCoherence/src')

# Import custom modules (uncomment as needed)
try:
    from preprocessing.filters import bandpass_filter, notch_filter
    from features.quantum_features import (extract_multiscale_entropy, 
                                         compute_nonlinear_scaling,
                                         phase_synchronization_analysis,
                                         wavelet_quantum_decomposition,
                                         nonlinear_transfer_entropy)
    modules_imported = True
    print("Successfully imported custom modules")
except ImportError as e:
    modules_imported = False
    print(f"Could not import some custom modules: {e}")
    print("Continuing with basic analysis")

# Define dataset-specific paths
dataset_name = 'necker_cube_eeg_Maksimenko'
project_root = '/blue/mcb4324/share/sdavidson2/QuantumNeuralCoherence'

# Create directory structure
raw_dir = os.path.join(project_root, 'data', 'raw', dataset_name)
processed_dir = os.path.join(project_root, 'data', 'processed', dataset_name)
results_dir = os.path.join(project_root, 'results', dataset_name)
figure_dir = os.path.join(results_dir, 'figures')
stats_dir = os.path.join(results_dir, 'statistics')
model_dir = os.path.join(results_dir, 'models')

# Create directories
for directory in [raw_dir, processed_dir, results_dir, figure_dir, stats_dir, model_dir]:
    os.makedirs(directory, exist_ok=True)

# Path to the data file
data_path = os.path.join(project_root, 'data', 'raw', 'EEG_data.mat')

# Check if the raw data file exists
if not os.path.exists(data_path):
    print(f"Raw data file not found at: {data_path}")
    print(f"Please ensure the file is located at this path")
    sys.exit(1)

# Create a README file in the dataset directory if it doesn't exist
readme_path = os.path.join(raw_dir, 'README.md')
if not os.path.exists(readme_path):
    with open(readme_path, 'w') as f:
        f.write(f"# {dataset_name}\n\n")
        f.write("## Dataset Information\n\n")
        f.write("- **Source**: Maksimenko et al.\n")
        f.write("- **Description**: EEG recordings during Necker cube perception with varying ambiguity levels\n")
        f.write("- **Participants**: 20 subjects\n")
        f.write("- **Channels**: 31 EEG channels\n")
        f.write("- **Sampling Rate**: 250 Hz\n")
        f.write("- **Conditions**: 8 ambiguity levels (0.15, 0.25, 0.4, 0.45, 0.55, 0.6, 0.75, 0.85)\n")
        f.write("- **Date Downloaded**: March 16, 2025\n")
    print(f"Created README file at {readme_path}")

# Load the data
print(f"Loading data from {data_path}...")
import scipy.io
mat_data = scipy.io.loadmat(data_path)
eeg_data = mat_data['EEG_data'][0]  # Get all subjects
print(f"Dataset contains data from {len(eeg_data)} subjects")

# Process subject data
def process_subject(subject_idx):
    """Process data for a single subject"""
    print(f"\nProcessing Subject {subject_idx+1}...")
    subject = eeg_data[subject_idx]
    
    # Extract key data
    trial_data = subject['trial'][0, 0]
    ambiguity_data = subject['Ambiguity'][0, 0]
    reaction_times = subject['ReactionTime'][0, 0]
    button_data = subject['Button'][0, 0]
    presentation_moments = subject['PresentationMoment'][0, 0]
    
    # Extract ambiguity values
    ambiguity_values = []
    for i in range(ambiguity_data.shape[1]):
        value = ambiguity_data[0, i]
        if isinstance(value, np.ndarray) and value.size > 0:
            ambiguity_values.append(float(value[0, 0]))
    
    # Extract button presses
    button_values = []
    for i in range(button_data.shape[1]):
        value = button_data[0, i]
        if isinstance(value, np.ndarray) and value.size > 0:
            button_values.append(int(value[0, 0]))
    
    # Extract reaction times
    rt_values = []
    for i in range(reaction_times.shape[1]):
        value = reaction_times[0, i]
        if isinstance(value, np.ndarray) and value.size > 0:
            rt_values.append(float(value[0, 0]))
    
    # Extract presentation moments
    pres_moments = []
    for i in range(presentation_moments.shape[1]):
        value = presentation_moments[0, i]
        if isinstance(value, np.ndarray) and value.size > 0:
            pres_moments.append(int(value[0, 0]))
    
    # Get trial data
    trials = []
    for i in range(trial_data.shape[1]):
        trial = trial_data[0, i]
        trials.append(trial)
    
    print(f"  Number of trials: {len(trials)}")
    print(f"  Number of channels: {trials[0].shape[0]}")
    print(f"  Timepoints per trial: {trials[0].shape[1]}")
    print(f"  Unique ambiguity levels: {sorted(set(ambiguity_values))}")
    
    # Organize trials by ambiguity level
    high_ambiguity_trials = []
    med_ambiguity_trials = []
    low_ambiguity_trials = []
    
    # Define ambiguity thresholds
    high_ambiguity_threshold = 0.5  # More ambiguous
    low_ambiguity_threshold = 0.3   # Less ambiguous
    
    # Categorize trials
    for i in range(len(trials)):
        trial = trials[i]
        ambiguity = ambiguity_values[i]
        button = button_values[i]
        rt = rt_values[i]
        
        if ambiguity > high_ambiguity_threshold:
            high_ambiguity_trials.append((i, trial, ambiguity, button, rt))
        elif ambiguity < low_ambiguity_threshold:
            low_ambiguity_trials.append((i, trial, ambiguity, button, rt))
        else:
            med_ambiguity_trials.append((i, trial, ambiguity, button, rt))
    
    print(f"  Found {len(high_ambiguity_trials)} high ambiguity trials")
    print(f"  Found {len(med_ambiguity_trials)} medium ambiguity trials")
    print(f"  Found {len(low_ambiguity_trials)} low ambiguity trials")
    
    # Save trial categorization for this subject
    trial_categories = {
        'subject_idx': subject_idx,
        'ambiguity_values': ambiguity_values,
        'button_values': button_values,
        'reaction_times': rt_values,
        'presentation_moments': pres_moments,
        'high_ambiguity_indices': [t[0] for t in high_ambiguity_trials],
        'med_ambiguity_indices': [t[0] for t in med_ambiguity_trials],
        'low_ambiguity_indices': [t[0] for t in low_ambiguity_trials]
    }
    
    categories_file = os.path.join(processed_dir, f'subject{subject_idx+1}_trial_categories.npy')
    np.save(categories_file, trial_categories)
    print(f"  Saved trial categorization to {categories_file}")
    
    return trials, ambiguity_values, button_values, rt_values, pres_moments, high_ambiguity_trials, low_ambiguity_trials

def compare_high_low_ambiguity(subject_idx, trials, high_ambiguity_trials, low_ambiguity_trials):
    """Compare EEG activity between high and low ambiguity trials"""
    print("\nComparing high vs. low ambiguity neural dynamics...")
    
    # Extract EEG data for high and low ambiguity trials
    high_amb_eeg = np.array([t[1] for t in high_ambiguity_trials])
    low_amb_eeg = np.array([t[1] for t in low_ambiguity_trials])
    
    # Calculate average across trials
    high_amb_avg = np.mean(high_amb_eeg, axis=0)
    low_amb_avg = np.mean(low_amb_eeg, axis=0)
    
    # Plot EEG for selected channels
    # Choose a diverse set of channels covering different brain regions
    channels_to_plot = [0, 2, 4, 6, 8]  # Modify based on actual channel layout
    
    plt.figure(figsize=(15, 10))
    fs = 250  # Sampling rate
    time = np.arange(high_amb_avg.shape[1]) / fs
    
    for i, ch in enumerate(channels_to_plot):
        plt.subplot(len(channels_to_plot), 1, i+1)
        plt.plot(time, high_amb_avg[ch], 'r', label='High Ambiguity')
        plt.plot(time, low_amb_avg[ch], 'b', label='Low Ambiguity')
        plt.title(f'Channel {ch+1}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.axvline(x=2.0, color='k', linestyle='--', label='Stimulus')  # Assuming stimulus at 2s
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    figure_path = os.path.join(figure_dir, f'subject{subject_idx+1}_high_vs_low_ambiguity_eeg.png')
    plt.savefig(figure_path)
    print(f"  Saved EEG comparison to {figure_path}")
    
    # Time-frequency analysis
    print("  Performing time-frequency analysis...")
    channel_idx = 0  # First channel (adjust as needed)
    
    # Calculate spectrograms for high ambiguity trials
    high_spectrograms = []
    for _, trial, _, _, _ in high_ambiguity_trials[:5]:  # Use first 5 trials
        f, t, Sxx = signal.spectrogram(trial[channel_idx], fs, nperseg=128, noverlap=64)
        high_spectrograms.append(10 * np.log10(Sxx))
    
    # Calculate spectrograms for low ambiguity trials
    low_spectrograms = []
    for _, trial, _, _, _ in low_ambiguity_trials[:5]:  # Use first 5 trials
        f, t, Sxx = signal.spectrogram(trial[channel_idx], fs, nperseg=128, noverlap=64)
        low_spectrograms.append(10 * np.log10(Sxx))
    
    # Average spectrograms
    high_spec_avg = np.mean(high_spectrograms, axis=0) if high_spectrograms else None
    low_spec_avg = np.mean(low_spectrograms, axis=0) if low_spectrograms else None
    
    if high_spec_avg is not None and low_spec_avg is not None:
        # Plot spectrograms
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        plt.pcolormesh(t, f, high_spec_avg, shading='gouraud')
        plt.title('High Ambiguity Trials - Average Spectrogram')
        plt.ylabel('Frequency [Hz]')
        plt.colorbar(label='Power/Frequency [dB/Hz]')
        plt.ylim(0, 50)
        
        plt.subplot(2, 1, 2)
        plt.pcolormesh(t, f, low_spec_avg, shading='gouraud')
        plt.title('Low Ambiguity Trials - Average Spectrogram')
        plt.xlabel('Time [sec]')
        plt.ylabel('Frequency [Hz]')
        plt.colorbar(label='Power/Frequency [dB/Hz]')
        plt.ylim(0, 50)
        
        plt.tight_layout()
        spec_path = os.path.join(figure_dir, f'subject{subject_idx+1}_high_vs_low_ambiguity_spectrogram.png')
        plt.savefig(spec_path)
        print(f"  Saved spectrogram comparison to {spec_path}")
        
        # Calculate the difference between high and low ambiguity spectrograms
        spec_diff = high_spec_avg - low_spec_avg
        
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(t, f, spec_diff, shading='gouraud', cmap='seismic')
        plt.title('Difference: High - Low Ambiguity Spectrograms')
        plt.xlabel('Time [sec]')
        plt.ylabel('Frequency [Hz]')
        plt.colorbar(label='Power Difference [dB/Hz]')
        plt.ylim(0, 50)
        
        plt.tight_layout()
        diff_path = os.path.join(figure_dir, f'subject{subject_idx+1}_ambiguity_spectrogram_difference.png')
        plt.savefig(diff_path)
        print(f"  Saved spectrogram difference to {diff_path}")
    
    return high_amb_avg, low_amb_avg, f, t, high_spec_avg, low_spec_avg

def analyze_quantum_signatures(subject_idx, trials, high_ambiguity_trials, low_ambiguity_trials):
    """Analyze potential quantum signatures in neural dynamics"""
    
    if not modules_imported:
        print("\nSkipping quantum signature analysis - custom modules not available")
        return
    
    print("\nAnalyzing potential quantum neural signatures...")
    
    # Select representative trials
    high_trial = high_ambiguity_trials[0][1]  # First high ambiguity trial
    low_trial = low_ambiguity_trials[0][1]    # First low ambiguity trial
    
    # Sample channels for analysis
    central_channel = 0      # e.g., Oz or O1
    frontal_channel = 6      # e.g., Fz or F3
    
    # 1. Multiscale Entropy Analysis
    try:
        print("  Computing multiscale entropy...")
        high_mse = extract_multiscale_entropy(high_trial[central_channel], m=2, r=0.15, scale_range=10)
        low_mse = extract_multiscale_entropy(low_trial[central_channel], m=2, r=0.15, scale_range=10)
        
        # Plot MSE results
        plt.figure(figsize=(10, 6))
        scales = range(1, len(high_mse) + 1)
        plt.plot(scales, high_mse, 'ro-', label='High Ambiguity')
        plt.plot(scales, low_mse, 'bo-', label='Low Ambiguity')
        plt.xlabel('Scale Factor')
        plt.ylabel('Sample Entropy')
        plt.title('Multiscale Entropy: High vs. Low Ambiguity')
        plt.legend()
        plt.grid(True)
        
        mse_path = os.path.join(figure_dir, f'subject{subject_idx+1}_multiscale_entropy.png')
        plt.savefig(mse_path)
        print(f"  Saved multiscale entropy analysis to {mse_path}")
        
        # Save MSE data
        mse_data = {
            'high_ambiguity_mse': high_mse,
            'low_ambiguity_mse': low_mse,
            'scales': scales
        }
        np.save(os.path.join(stats_dir, f'subject{subject_idx+1}_mse_data.npy'), mse_data)
    except Exception as e:
        print(f"  Error in multiscale entropy analysis: {e}")
    
    # 2. Phase Synchronization Analysis
    try:
        print("  Computing phase synchronization...")
        # Calculate phase synchronization between frontal and central regions
        fs = 250  # Sampling rate
        
        # Alpha band (8-13 Hz)
        alpha_band = (8, 13)
        high_psi_alpha, high_entropy_alpha = phase_synchronization_analysis(
            high_trial[frontal_channel], high_trial[central_channel], fs, alpha_band
        )
        
        low_psi_alpha, low_entropy_alpha = phase_synchronization_analysis(
            low_trial[frontal_channel], low_trial[central_channel], fs, alpha_band
        )
        
        # Beta band (13-30 Hz)
        beta_band = (13, 30)
        high_psi_beta, high_entropy_beta = phase_synchronization_analysis(
            high_trial[frontal_channel], high_trial[central_channel], fs, beta_band
        )
        
        low_psi_beta, low_entropy_beta = phase_synchronization_analysis(
            low_trial[frontal_channel], low_trial[central_channel], fs, beta_band
        )
        
        # Plot results
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        bands = ['Alpha', 'Beta']
        high_psi = [high_psi_alpha, high_psi_beta]
        low_psi = [low_psi_alpha, low_psi_beta]
        
        x = range(len(bands))
        width = 0.35
        plt.bar([i - width/2 for i in x], high_psi, width, label='High Ambiguity')
        plt.bar([i + width/2 for i in x], low_psi, width, label='Low Ambiguity')
        plt.xlabel('Frequency Band')
        plt.ylabel('Phase Synchronization Index (PSI)')
        plt.title('Phase Synchronization by Frequency Band')
        plt.xticks(x, bands)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        high_entropy = [high_entropy_alpha, high_entropy_beta]
        low_entropy = [low_entropy_alpha, low_entropy_beta]
        
        plt.bar([i - width/2 for i in x], high_entropy, width, label='High Ambiguity')
        plt.bar([i + width/2 for i in x], low_entropy, width, label='Low Ambiguity')
        plt.xlabel('Frequency Band')
        plt.ylabel('Phase Difference Entropy')
        plt.title('Phase Entropy by Frequency Band')
        plt.xticks(x, bands)
        plt.legend()
        
        plt.tight_layout()
        sync_path = os.path.join(figure_dir, f'subject{subject_idx+1}_phase_synchronization.png')
        plt.savefig(sync_path)
        print(f"  Saved phase synchronization analysis to {sync_path}")
        
        # Save synchronization data
        sync_data = {
            'high_psi_alpha': high_psi_alpha,
            'low_psi_alpha': low_psi_alpha,
            'high_psi_beta': high_psi_beta,
            'low_psi_beta': low_psi_beta,
            'high_entropy_alpha': high_entropy_alpha,
            'low_entropy_alpha': low_entropy_alpha,
            'high_entropy_beta': high_entropy_beta,
            'low_entropy_beta': low_entropy_beta
        }
        np.save(os.path.join(stats_dir, f'subject{subject_idx+1}_sync_data.npy'), sync_data)
    except Exception as e:
        print(f"  Error in phase synchronization analysis: {e}")
    
    # 3. Transfer Entropy Analysis (information flow)
    try:
        print("  Computing transfer entropy...")
        # Calculate transfer entropy between frontal and central regions
        high_te_fc, high_sig_fc = nonlinear_transfer_entropy(
            high_trial[frontal_channel], high_trial[central_channel]
        )
        
        high_te_cf, high_sig_cf = nonlinear_transfer_entropy(
            high_trial[central_channel], high_trial[frontal_channel]
        )
        
        low_te_fc, low_sig_fc = nonlinear_transfer_entropy(
            low_trial[frontal_channel], low_trial[central_channel]
        )
        
        low_te_cf, low_sig_cf = nonlinear_transfer_entropy(
            low_trial[central_channel], low_trial[frontal_channel]
        )
        
        # Plot results
        plt.figure(figsize=(10, 6))
        
        directions = ['Frontal → Central', 'Central → Frontal']
        high_te = [high_te_fc, high_te_cf]
        low_te = [low_te_fc, low_te_cf]
        
        x = range(len(directions))
        width = 0.35
        plt.bar([i - width/2 for i in x], high_te, width, label='High Ambiguity')
        plt.bar([i + width/2 for i in x], low_te, width, label='Low Ambiguity')
        plt.xlabel('Direction')
        plt.ylabel('Transfer Entropy (bits)')
        plt.title('Information Flow Between Brain Regions')
        plt.xticks(x, directions)
        plt.legend()
        
        plt.tight_layout()
        te_path = os.path.join(figure_dir, f'subject{subject_idx+1}_transfer_entropy.png')
        plt.savefig(te_path)
        print(f"  Saved transfer entropy analysis to {te_path}")
        
        # Save transfer entropy data
        te_data = {
            'high_te_fc': high_te_fc,
            'high_te_cf': high_te_cf,
            'low_te_fc': low_te_fc,
            'low_te_cf': low_te_cf,
            'high_sig_fc': high_sig_fc,
            'high_sig_cf': high_sig_cf,
            'low_sig_fc': low_sig_fc,
            'low_sig_cf': low_sig_cf
        }
        np.save(os.path.join(stats_dir, f'subject{subject_idx+1}_te_data.npy'), te_data)
    except Exception as e:
        print(f"  Error in transfer entropy analysis: {e}")

def main():
    """Main analysis function"""
    # Process data for the first subject
    subject_idx = 0
    
    print(f"Starting analysis for Subject {subject_idx+1}...")
    
    # Extract and organize data
    trials, ambiguity_values, button_values, rt_values, pres_moments, high_ambiguity_trials, low_ambiguity_trials = process_subject(subject_idx)
    
    # Basic EEG and time-frequency analysis
    high_amb_avg, low_amb_avg, freq, time, high_spec_avg, low_spec_avg = compare_high_low_ambiguity(
        subject_idx, trials, high_ambiguity_trials, low_ambiguity_trials
    )
    
    # Advanced quantum signatures analysis
    analyze_quantum_signatures(
        subject_idx, trials, high_ambiguity_trials, low_ambiguity_trials
    )
    
    print("\nAnalysis completed successfully!")
    print(f"Results saved to: {results_dir}")

if __name__ == "__main__":
    main()
