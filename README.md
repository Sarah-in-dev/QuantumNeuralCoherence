Quantum Neural Coherence Detection Project: Necker Cube EEG Analysis
Project Status Update (March 16, 2025)
This document provides a comprehensive overview of our recent integration and analysis of the Necker Cube EEG dataset (Maksimenko et al.), including detailed documentation of the implementation, current progress, and future plans.
Project Overview
The Quantum Neural Coherence Detection Project aims to identify potential quantum signatures in neural dynamics during perceptual decision-making. We're investigating whether quantum effects might play a role in neural information processing by looking for signatures of quantum coherence, entanglement, and non-local interactions in EEG data.
We've recently integrated a dataset of EEG recordings during perceptual bistability tasks with varying levels of ambiguity (Necker Cube experiment), making it ideal for investigating quantum neural signatures during perceptual uncertainty.
Dataset Description
The Necker Cube EEG dataset contains:

EEG recordings from 20 healthy subjects
Visual stimuli classification task using Necker cubes with varying ambiguity
8 ambiguity levels: {0.0, 0.15, 0.4, 0.45, 0.55, 0.6, 0.85, 1}
80 trials per subject (40 from beginning of experiment, 40 from end)
31 EEG channels sampled at 250 Hz
4-second recordings per trial with stimulus presented mid-trial
Button press responses (1=left, 2=right orientation) and reaction times

Implementation Details
1. Dataset Acquisition (March 16, 2025)
Downloaded and extracted the dataset:
wget -P raw/ https://figshare.com/ndownloader/articles/12155343/versions/2
mv raw/2 raw/necker_cube_eeg_dataset_Maksimenko.zip
unzip raw/necker_cube_eeg_dataset_Maksimenko.zip

2. Directory Structure Implementation
Created an organized, scalable directory structure for multi-dataset analysis:

QuantumNeuralCoherence/
├── data/
│   ├── raw/
│   │   ├── necker_cube_eeg_Maksimenko/
│   │   └── [other_datasets]/
│   └── processed/
│       ├── necker_cube_eeg_Maksimenko/
│       └── [other_datasets]/
├── src/
│   ├── preprocessing/
│   │   └── filters.py
│   ├── features/
│   │   └── quantum_features.py
│   ├── models/
│   └── visualization/
├── code/
│   ├── preprocessing/
│   ├── analysis/
│   │   └── analyze_necker_eeg.py
│   └── visualization/
├── results/
│   ├── necker_cube_eeg_Maksimenko/
│   │   ├── figures/
│   │   ├── statistics/
│   │   └── models/
│   └── [other_datasets]/
└── docs/

3. Analysis Script Development
Created analyze_necker_eeg.py with the following components:
Data Loading and Organization

# Load the data
mat_data = scipy.io.loadmat(data_path)
eeg_data = mat_data['EEG_data'][0]  # Get all subjects

# Extract data for a subject
subject = eeg_data[subject_idx]
trial_data = subject['trial'][0, 0]
ambiguity_data = subject['Ambiguity'][0, 0]
reaction_times = subject['ReactionTime'][0, 0]
button_data = subject['Button'][0, 0]
presentation_moments = subject['PresentationMoment'][0, 0]

Trial Categorization

# Categorize trials by ambiguity level
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
    
    if ambiguity > high_ambiguity_threshold:
        high_ambiguity_trials.append((i, trial, ambiguity, button, rt))
    elif ambiguity < low_ambiguity_threshold:
        low_ambiguity_trials.append((i, trial, ambiguity, button, rt))
    else:
        med_ambiguity_trials.append((i, trial, ambiguity, button, rt))

EEG Analysis

# Calculate average across trials
high_amb_avg = np.mean(high_amb_eeg, axis=0)
low_amb_avg = np.mean(low_amb_eeg, axis=0)

# Plot EEG for selected channels
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

Time-Frequency Analysis

# Calculate spectrograms
f, t, Sxx = signal.spectrogram(trial[channel_idx], fs, nperseg=128, noverlap=64)
high_spectrograms.append(10 * np.log10(Sxx))

# Calculate the difference between high and low ambiguity spectrograms
spec_diff = high_spec_avg - low_spec_avg

plt.figure(figsize=(10, 6))
plt.pcolormesh(t, f, spec_diff, shading='gouraud', cmap='seismic')
plt.title('Difference: High - Low Ambiguity Spectrograms')
plt.xlabel('Time [sec]')
plt.ylabel('Frequency [Hz]')

4. Quantum Signature Analysis Implementation
Implemented functions to analyze potential quantum signatures in neural dynamics:
Multiscale Entropy Analysis

high_mse = extract_multiscale_entropy(high_trial[central_channel], m=2, r=0.15, scale_range=10)
low_mse = extract_multiscale_entropy(low_trial[central_channel], m=2, r=0.15, scale_range=10)

Phase Synchronization Analysis

high_psi_alpha, high_entropy_alpha = phase_synchronization_analysis(
    high_trial[frontal_channel], high_trial[central_channel], fs, alpha_band
)

low_psi_alpha, low_entropy_alpha = phase_synchronization_analysis(
    low_trial[frontal_channel], low_trial[central_channel], fs, alpha_band
)

Transfer Entropy Analysis

high_te_fc, high_sig_fc = nonlinear_transfer_entropy(
    high_trial[frontal_channel], high_trial[central_channel]
)

low_te_fc, low_sig_fc = nonlinear_transfer_entropy(
    low_trial[frontal_channel], low_trial[central_channel]
)

Current Progress
Completed Work (March 16, 2025)

Dataset Integration:

Successfully downloaded and organized the Necker Cube EEG dataset
Created a scalable directory structure for multi-dataset analysis
Generated README with dataset metadata


Basic Analysis Pipeline:

Extracted EEG data, ambiguity levels, button presses, and reaction times
Categorized trials by ambiguity level (48 high, 15 medium, 17 low ambiguity)
Generated EEG waveform comparisons between high and low ambiguity trials
Created time-frequency spectrograms and difference visualizations


Initial Findings:

Successfully processed Subject 1's data (out of 20 subjects)
Generated visualizations showing differences between high and low ambiguity conditions
Created a foundational framework for quantum neural signature analysis



Technical Challenges and Solutions

MATLAB Data Structure Navigation:

Challenge: The dataset used nested MATLAB cell arrays requiring careful extraction
Solution: Implemented systematic extraction of each data component


Module Import Issue:

Challenge: Custom modules not found in Python path
Solution: Added graceful fallback to basic analysis when modules unavailable
Pending: Fix by adding project root to Python path or installing as package


File Organization:

Challenge: Needed scalable structure for multiple datasets
Solution: Implemented dataset-specific directory structure with consistent naming

Future Plans
Short-Term Plans (Next 2-4 Weeks)

Fix Module Import Issues:

import sys
sys.path.append('/blue/mcb4324/share/sdavidson2/QuantumNeuralCoherence')
# Then try importing
from src.preprocessing.filters import bandpass_filter

Extend Analysis to All Subjects:

def main():
    for subject_idx in range(20):  # 0-19 for 20 subjects
        print(f"\nStarting analysis for Subject {subject_idx+1}...")
        # Process subject data
        # ...

Implement Advanced Quantum Metrics:

Complete integration of quantum feature extraction modules
Apply multiscale entropy, phase synchronization, and transfer entropy analyses
Compare quantum vs. classical processing signatures


Group-Level Analysis:

Aggregate metrics across subjects
Implement statistical tests (t-tests, ANOVA, cluster-based permutation)
Identify consistent patterns across subjects



Medium-Term Plans (1-2 Months)

Additional Quantum Metrics:

Implement integrated information (Φ) measures
Develop non-local correlation detection algorithms
Analyze scale-free dynamics and critical phenomena


Neural-Behavioral Correlations:

Correlate neural signatures with reaction times and accuracy
Examine how quantum signatures relate to perceptual uncertainty
Develop predictive models of perceptual decisions


Theoretical Framework Development:

Compare empirical findings with predictions from quantum cognition models
Develop testable hypotheses for differentiating quantum vs. classical processing



Long-Term Goals (3+ Months)

Multi-Dataset Integration:

Apply analysis pipeline to additional perceptual bistability datasets
Compare findings across different experimental paradigms
Develop cross-dataset validation approaches


Advanced Mathematical Framework:

Develop more sophisticated mathematical models of quantum neural dynamics
Create simulation frameworks to validate detection methods
Connect with quantum field theories applicable to biological systems


Applications Development:

Explore applications for consciousness research
Investigate potential for brain-computer interfaces
Develop clinical applications for neurological disorders



Usage Instructions
Setting Up the Environment

# Create conda environment
conda create -n quantum_neuro python=3.9
conda activate quantum_neuro

# Install dependencies
conda install numpy scipy matplotlib pandas scikit-learn
conda install -c conda-forge mne
pip install -r requirements.txt

Running the Analysis

# Navigate to the analysis directory
cd /blue/mcb4324/share/sdavidson2/QuantumNeuralCoherence/code/analysis

# Run the analysis script
python analyze_necker_eeg.py

Accessing Results
The analysis generates several types of outputs:

Processed Data:

Trial categorization data (high/medium/low ambiguity)
Subject-specific statistics


Visualizations:

EEG waveform comparisons
Time-frequency spectrograms
Quantum signature metrics


Statistical Results:

Extracted features and metrics
Comparison statistics



References

Maksimenko, V.A., et al. (2020). "Dissociating cognitive processes during ambiguous information processing in perceptual decision-making." Frontiers in Behavioral Neuroscience, 14, 95.
Tononi, G., et al. (2016). "Integrated information theory: from consciousness to its physical substrate." Nature Reviews Neuroscience, 17(7), 450-461.
Fisher, M.P. (2015). "Quantum cognition: The possibility of processing with nuclear spins in the brain." Annals of Physics, 362, 593-602.
