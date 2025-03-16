import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal

# Path to the data file
data_path = '/blue/mcb4324/share/sdavidson2/QuantumNeuralCoherence/data/raw/EEG_data.mat'

# Create output directories
results_dir = '/blue/mcb4324/share/sdavidson2/QuantumNeuralCoherence/results'
figure_dir = os.path.join(results_dir, 'figures')
processed_dir = os.path.join(results_dir, 'processed')
os.makedirs(figure_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)

# Load the data
print("Loading data...")
mat_data = scipy.io.loadmat(data_path)
eeg_data = mat_data['EEG_data'][0]  # Get all subjects

# Process the first subject for initial analysis
subject_idx = 0
subject = eeg_data[subject_idx]

# Extract data
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

print(f"Subject {subject_idx+1} data:")
print(f"  Number of trials: {trial_data.shape[1]}")
print(f"  Number of ambiguity values: {len(ambiguity_values)}")
print(f"  Number of button presses: {len(button_values)}")
print(f"  Number of reaction times: {len(rt_values)}")
print(f"  Number of presentation moments: {len(pres_moments)}")

# Analyze time-frequency dynamics for high vs. low ambiguity trials
high_ambiguity_trials = []
low_ambiguity_trials = []

# Define high and low ambiguity
high_ambiguity_threshold = 0.5
low_ambiguity_threshold = 0.3

# Collect high and low ambiguity trials
for i in range(trial_data.shape[1]):
    trial = trial_data[0, i]
    ambiguity = ambiguity_values[i]
    button = button_values[i]
    rt = rt_values[i]
    
    if ambiguity > high_ambiguity_threshold:
        high_ambiguity_trials.append((i, trial, ambiguity, button, rt))
    elif ambiguity < low_ambiguity_threshold:
        low_ambiguity_trials.append((i, trial, ambiguity, button, rt))

print(f"\nFound {len(high_ambiguity_trials)} high ambiguity trials")
print(f"Found {len(low_ambiguity_trials)} low ambiguity trials")

# Plot average EEG for high vs. low ambiguity trials
if high_ambiguity_trials and low_ambiguity_trials:
    # Extract EEG data for high and low ambiguity trials
    high_amb_eeg = np.array([t[1] for t in high_ambiguity_trials])
    low_amb_eeg = np.array([t[1] for t in low_ambiguity_trials])
    
    # Calculate average across trials
    high_amb_avg = np.mean(high_amb_eeg, axis=0)
    low_amb_avg = np.mean(low_amb_eeg, axis=0)
    
    # Plot EEG for selected channels
    channels_to_plot = [0, 2, 4, 6, 8]  # Adjust based on actual channel locations
    
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
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, 'high_vs_low_ambiguity_eeg.png'))
    print(f"Saved high vs. low ambiguity EEG comparison to {os.path.join(figure_dir, 'high_vs_low_ambiguity_eeg.png')}")
    
    # Time-frequency analysis for a representative channel
    channel_idx = 0  # First channel (can be adjusted)
    
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
        plt.savefig(os.path.join(figure_dir, 'high_vs_low_ambiguity_spectrogram.png'))
        print(f"Saved spectrogram comparison to {os.path.join(figure_dir, 'high_vs_low_ambiguity_spectrogram.png')}")
        
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
        plt.savefig(os.path.join(figure_dir, 'ambiguity_spectrogram_difference.png'))
        print(f"Saved spectrogram difference to {os.path.join(figure_dir, 'ambiguity_spectrogram_difference.png')}")

# Save some processed data for further analysis
processed_data = {
    'subject_idx': subject_idx,
    'ambiguity_values': ambiguity_values,
    'button_values': button_values,
    'reaction_times': rt_values,
    'presentation_moments': pres_moments,
    'high_ambiguity_indices': [t[0] for t in high_ambiguity_trials],
    'low_ambiguity_indices': [t[0] for t in low_ambiguity_trials]
}

np.save(os.path.join(processed_dir, 'subject1_processed.npy'), processed_data)
print(f"Saved processed data to {os.path.join(processed_dir, 'subject1_processed.npy')}")

print("\nInitial analysis complete!")
