#!/usr/bin/env python
"""
Group-level analysis script for Necker Cube EEG dataset (Maksimenko et al.)
This script processes EEG data recorded during Necker cube perception with varying ambiguity levels,
and performs both single-subject and group-level analyses.

Author: Sarah Davidson
Date: March 22, 2025
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import warnings
import argparse
import pandas as pd
from tqdm import tqdm
import traceback
import logging
import json
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('necker_eeg_analysis.log')
    ]
)
logger = logging.getLogger('analyze_necker_eeg')

# Add the source directory to path to access custom modules
# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up to the project root directory
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)
logger.info(f"Added {project_root} to Python path")

# Import custom modules
try:
    from src.preprocessing.filters import bandpass_filter, notch_filter
    from src.features.quantum_features import (extract_multiscale_entropy, 
                                             compute_nonlinear_scaling,
                                             phase_synchronization_analysis,
                                             wavelet_quantum_decomposition,
                                             nonlinear_transfer_entropy)
    modules_imported = True
    logger.info("Successfully imported custom modules")
except ImportError as e:
    modules_imported = False
    logger.warning(f"Could not import some custom modules: {e}")
    logger.info("Continuing with basic analysis")

class DatasetProcessor:
    """Process and analyze the Necker Cube EEG dataset."""
    
    def __init__(self, data_path=None, output_base_dir=None):
        """
        Initialize the processor with data path and output directory.
        
        Parameters:
        -----------
        data_path : str
            Path to the EEG data file
        output_base_dir : str
            Base directory for saving results
        """
        # Default paths if not provided
        if data_path is None:
            self.data_path = os.path.join(project_root, 'data', 'raw', 'EEG_data.mat')
        else:
            self.data_path = data_path
            
        if output_base_dir is None:
            self.output_base_dir = os.path.join(project_root, 'results')
        else:
            self.output_base_dir = output_base_dir
            
        # Dataset-specific paths
        self.dataset_name = 'necker_cube_eeg_Maksimenko'
        self.raw_dir = os.path.join(project_root, 'data', 'raw', self.dataset_name)
        self.processed_dir = os.path.join(project_root, 'data', 'processed', self.dataset_name)
        self.results_dir = os.path.join(self.output_base_dir, self.dataset_name)
        self.figure_dir = os.path.join(self.results_dir, 'figures')
        self.stats_dir = os.path.join(self.results_dir, 'statistics')
        self.model_dir = os.path.join(self.results_dir, 'models')
        self.group_dir = os.path.join(self.results_dir, 'group_analysis')
        
        # Create directories
        for directory in [self.raw_dir, self.processed_dir, self.results_dir, 
                           self.figure_dir, self.stats_dir, self.model_dir, self.group_dir]:
            os.makedirs(directory, exist_ok=True)
            
        # Data containers
        self.eeg_data = None
        self.subjects_data = {}  # Store processed data for each subject
        self.group_metrics = {}  # Store group-level metrics
    
    def load_data(self):
        """Load the EEG dataset from MAT file."""
        try:
            logger.info(f"Loading data from {self.data_path}")
            import scipy.io
            mat_data = scipy.io.loadmat(self.data_path)
            self.eeg_data = mat_data['EEG_data'][0]  # Get all subjects
            logger.info(f"Dataset contains data from {len(self.eeg_data)} subjects")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def process_subject(self, subject_idx):
        """
        Process data for a single subject.
        
        Parameters:
        -----------
        subject_idx : int
            Index of the subject to process
        
        Returns:
        --------
        tuple
            Processed data: (trials, ambiguity_values, button_values, rt_values, 
                            pres_moments, high_ambiguity_trials, medium_ambiguity_trials, 
                            low_ambiguity_trials)
        """
        try:
            logger.info(f"Processing Subject {subject_idx+1}...")
            subject = self.eeg_data[subject_idx]
            
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
            
            logger.info(f"  Number of trials: {len(trials)}")
            logger.info(f"  Number of channels: {trials[0].shape[0]}")
            logger.info(f"  Timepoints per trial: {trials[0].shape[1]}")
            logger.info(f"  Unique ambiguity levels: {sorted(set(ambiguity_values))}")
            
            # Organize trials by ambiguity level
            high_ambiguity_trials = []
            med_ambiguity_trials = []
            low_ambiguity_trials = []
            
            # Define ambiguity thresholds
            high_ambiguity_threshold = 0.5  # More ambiguous
            low_ambiguity_threshold = 0.3   # Less ambiguous
            
            # Categorize trials
            for i in range(len(trials)):
                if i >= len(ambiguity_values) or i >= len(button_values) or i >= len(rt_values):
                    logger.warning(f"  Trial {i} has missing metadata, skipping")
                    continue
                    
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
            
            logger.info(f"  Found {len(high_ambiguity_trials)} high ambiguity trials")
            logger.info(f"  Found {len(med_ambiguity_trials)} medium ambiguity trials")
            logger.info(f"  Found {len(low_ambiguity_trials)} low ambiguity trials")
            
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
            
            categories_file = os.path.join(self.processed_dir, f'subject{subject_idx+1}_trial_categories.npy')
            np.save(categories_file, trial_categories)
            logger.info(f"  Saved trial categorization to {categories_file}")
            
            # Store in the subjects_data dictionary
            self.subjects_data[subject_idx] = {
                'trials': trials,
                'ambiguity_values': ambiguity_values,
                'button_values': button_values,
                'reaction_times': rt_values,
                'presentation_moments': pres_moments,
                'high_ambiguity_trials': high_ambiguity_trials,
                'med_ambiguity_trials': med_ambiguity_trials,
                'low_ambiguity_trials': low_ambiguity_trials
            }
            
            return (trials, ambiguity_values, button_values, rt_values, 
                   pres_moments, high_ambiguity_trials, med_ambiguity_trials, 
                   low_ambiguity_trials)
        
        except Exception as e:
            logger.error(f"Error processing subject {subject_idx+1}: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def compare_high_low_ambiguity(self, subject_idx, trials, high_ambiguity_trials, low_ambiguity_trials):
        """
        Compare EEG activity between high and low ambiguity trials for a subject.
        
        Parameters:
        -----------
        subject_idx : int
            Index of the subject
        trials : list
            List of all trials
        high_ambiguity_trials : list
            List of high ambiguity trials
        low_ambiguity_trials : list
            List of low ambiguity trials
        
        Returns:
        --------
        tuple
            Analysis results: (high_amb_avg, low_amb_avg, f, t, high_spec_avg, low_spec_avg)
        """
        try:
            logger.info(f"Comparing high vs. low ambiguity neural dynamics for Subject {subject_idx+1}...")
            
            # Check if we have enough trials for comparison
            if len(high_ambiguity_trials) < 3 or len(low_ambiguity_trials) < 3:
                logger.warning(f"  Not enough trials for reliable comparison: {len(high_ambiguity_trials)} high, {len(low_ambiguity_trials)} low")
                return None
            
            # Extract EEG data for high and low ambiguity trials
            high_amb_eeg = np.array([t[1] for t in high_ambiguity_trials])
            low_amb_eeg = np.array([t[1] for t in low_ambiguity_trials])
            
            # Calculate average across trials
            high_amb_avg = np.mean(high_amb_eeg, axis=0)
            low_amb_avg = np.mean(low_amb_eeg, axis=0)
            
            # Plot EEG for selected channels
            # Choose a diverse set of channels covering different brain regions
            channels_to_plot = [0, 5, 10, 15, 20]  # Modify based on actual channel layout
            
            plt.figure(figsize=(15, 10))
            fs = 250  # Sampling rate
            time = np.arange(high_amb_avg.shape[1]) / fs
            
            for i, ch in enumerate(channels_to_plot):
                if ch >= high_amb_avg.shape[0]:
                    continue  # Skip if channel index is out of range
                
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
            figure_path = os.path.join(self.figure_dir, f'subject{subject_idx+1}_high_vs_low_ambiguity_eeg.png')
            plt.savefig(figure_path)
            plt.close()
            logger.info(f"  Saved EEG comparison to {figure_path}")
            
            # Time-frequency analysis
            logger.info("  Performing time-frequency analysis...")
            channel_idx = 0  # First channel (adjust as needed)
            
            # Calculate spectrograms for high ambiguity trials
            high_spectrograms = []
            for _, trial, _, _, _ in high_ambiguity_trials[:min(5, len(high_ambiguity_trials))]:  # Use up to 5 trials
                f, t, Sxx = signal.spectrogram(trial[channel_idx], fs, nperseg=128, noverlap=64)
                high_spectrograms.append(10 * np.log10(Sxx + 1e-10))  # Add small value to avoid log(0)
            
            # Calculate spectrograms for low ambiguity trials
            low_spectrograms = []
            for _, trial, _, _, _ in low_ambiguity_trials[:min(5, len(low_ambiguity_trials))]:  # Use up to 5 trials
                f, t, Sxx = signal.spectrogram(trial[channel_idx], fs, nperseg=128, noverlap=64)
                low_spectrograms.append(10 * np.log10(Sxx + 1e-10))  # Add small value to avoid log(0)
            
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
                spec_path = os.path.join(self.figure_dir, f'subject{subject_idx+1}_high_vs_low_ambiguity_spectrogram.png')
                plt.savefig(spec_path)
                plt.close()
                logger.info(f"  Saved spectrogram comparison to {spec_path}")
                
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
                diff_path = os.path.join(self.figure_dir, f'subject{subject_idx+1}_ambiguity_spectrogram_difference.png')
                plt.savefig(diff_path)
                plt.close()
                logger.info(f"  Saved spectrogram difference to {diff_path}")
                
                # Calculate and save frequency band powers for group analysis
                bands = {
                    'delta': (1, 4),
                    'theta': (4, 8),
                    'alpha': (8, 13),
                    'beta': (13, 30),
                    'gamma': (30, 45)
                }
                
                band_powers = {}
                for band_name, (fmin, fmax) in bands.items():
                    # Find indices for this frequency band
                    idx = np.logical_and(f >= fmin, f <= fmax)
                    
                    # Calculate average power in band for high and low ambiguity
                    high_power = np.mean(high_spec_avg[idx, :])
                    low_power = np.mean(low_spec_avg[idx, :])
                    diff_power = high_power - low_power
                    
                    band_powers[band_name] = {
                        'high_ambiguity': float(high_power),
                        'low_ambiguity': float(low_power),
                        'difference': float(diff_power)
                    }
                
                # Save band powers for this subject
                band_powers_file = os.path.join(self.stats_dir, f'subject{subject_idx+1}_band_powers.json')
                with open(band_powers_file, 'w') as f:
                    json.dump(band_powers, f, indent=4)
                logger.info(f"  Saved frequency band powers to {band_powers_file}")
                
                # Store results in subjects_data
                if subject_idx in self.subjects_data:
                    self.subjects_data[subject_idx]['band_powers'] = band_powers
            
            return high_amb_avg, low_amb_avg, f, t, high_spec_avg, low_spec_avg
        
        except Exception as e:
            logger.error(f"Error comparing high vs. low ambiguity for subject {subject_idx+1}: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def analyze_quantum_signatures(self, subject_idx, trials, high_ambiguity_trials, low_ambiguity_trials):
        """
        Analyze potential quantum signatures in neural dynamics.
        
        Parameters:
        -----------
        subject_idx : int
            Index of the subject
        trials : list
            List of all trials
        high_ambiguity_trials : list
            List of high ambiguity trials
        low_ambiguity_trials : list
            List of low ambiguity trials
        
        Returns:
        --------
        dict
            Quantum metrics for the subject
        """
        if not modules_imported:
            logger.warning(f"Skipping quantum signature analysis for Subject {subject_idx+1} - custom modules not available")
            return None
        
        if not high_ambiguity_trials or not low_ambiguity_trials:
            logger.warning(f"Skipping quantum signature analysis for Subject {subject_idx+1} - not enough trials")
            return None
        
        try:
            logger.info(f"Analyzing potential quantum neural signatures for Subject {subject_idx+1}...")
            
            # Select representative trials
            high_trial = high_ambiguity_trials[0][1]  # First high ambiguity trial
            low_trial = low_ambiguity_trials[0][1]    # First low ambiguity trial
            
            # Sample channels for analysis
            central_channel = 0      # e.g., Oz or O1
            frontal_channel = 6      # e.g., Fz or F3
            
            # Container for quantum metrics
            quantum_metrics = {
                'subject_idx': subject_idx,
                'multiscale_entropy': {},
                'phase_synchronization': {},
                'transfer_entropy': {}
            }
            
            # 1. Multiscale Entropy Analysis
            try:
                logger.info("  Computing multiscale entropy...")
                high_mse = extract_multiscale_entropy(high_trial[central_channel], m=2, r=0.15, scale_range=10)
                low_mse = extract_multiscale_entropy(low_trial[central_channel], m=2, r=0.15, scale_range=10)
                
                # Store metrics
                quantum_metrics['multiscale_entropy'] = {
                    'high_ambiguity': [float(x) for x in high_mse],
                    'low_ambiguity': [float(x) for x in low_mse],
                    'difference': [float(h - l) for h, l in zip(high_mse, low_mse)]
                }
                
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
                
                mse_path = os.path.join(self.figure_dir, f'subject{subject_idx+1}_multiscale_entropy.png')
                plt.savefig(mse_path)
                plt.close()
                logger.info(f"  Saved multiscale entropy analysis to {mse_path}")
                
                # Save MSE data
                mse_data = {
                    'high_ambiguity_mse': [float(x) for x in high_mse],
                    'low_ambiguity_mse': [float(x) for x in low_mse],
                    'scales': list(scales)
                }
                np.save(os.path.join(self.stats_dir, f'subject{subject_idx+1}_mse_data.npy'), mse_data)
            
            except Exception as e:
                logger.warning(f"  Error in multiscale entropy analysis: {e}")
            
            # 2. Phase Synchronization Analysis
            try:
                logger.info("  Computing phase synchronization...")
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
                
                # Store metrics
                quantum_metrics['phase_synchronization'] = {
                    'alpha': {
                        'high_ambiguity_psi': float(high_psi_alpha),
                        'low_ambiguity_psi': float(low_psi_alpha),
                        'high_ambiguity_entropy': float(high_entropy_alpha),
                        'low_ambiguity_entropy': float(low_entropy_alpha),
                        'psi_difference': float(high_psi_alpha - low_psi_alpha),
                        'entropy_difference': float(high_entropy_alpha - low_entropy_alpha)
                    },
                    'beta': {
                        'high_ambiguity_psi': float(high_psi_beta),
                        'low_ambiguity_psi': float(low_psi_beta),
                        'high_ambiguity_entropy': float(high_entropy_beta),
                        'low_ambiguity_entropy': float(low_entropy_beta),
                        'psi_difference': float(high_psi_beta - low_psi_beta),
                        'entropy_difference': float(high_entropy_beta - low_entropy_beta)
                    }
                }
                
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
                sync_path = os.path.join(self.figure_dir, f'subject{subject_idx+1}_phase_synchronization.png')
                plt.savefig(sync_path)
                plt.close()
                logger.info(f"  Saved phase synchronization analysis to {sync_path}")
                
                # Save synchronization data
                sync_data = {
                    'high_psi_alpha': float(high_psi_alpha),
                    'low_psi_alpha': float(low_psi_alpha),
                    'high_psi_beta': float(high_psi_beta),
                    'low_psi_beta': float(low_psi_beta),
                    'high_entropy_alpha': float(high_entropy_alpha),
                    'low_entropy_alpha': float(low_entropy_alpha),
                    'high_entropy_beta': float(high_entropy_beta),
                    'low_entropy_beta': float(low_entropy_beta)
                }
                np.save(os.path.join(self.stats_dir, f'subject{subject_idx+1}_sync_data.npy'), sync_data)
            
            except Exception as e:
                logger.warning(f"  Error in phase synchronization analysis: {e}")
                logger.warning(traceback.format_exc())
            
            # 3. Transfer Entropy Analysis (information flow)
            try:
                logger.info("  Computing transfer entropy...")
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
                
                # Store metrics
                quantum_metrics['transfer_entropy'] = {
                    'frontal_to_central': {
                        'high_ambiguity': float(high_te_fc),
                        'high_significance': float(high_sig_fc),
                        'low_ambiguity': float(low_te_fc),
                        'low_significance': float(low_sig_fc),
                        'difference': float(high_te_fc - low_te_fc)
                    },
                    'central_to_frontal': {
                        'high_ambiguity': float(high_te_cf),
                        'high_significance': float(high_sig_cf),
                        'low_ambiguity': float(low_te_cf),
                        'low_significance': float(low_sig_cf),
                        'difference': float(high_te_cf - low_te_cf)
                    }
                }
                
                # Calculate asymmetry (directionality)
                quantum_metrics['transfer_entropy']['asymmetry'] = {
                    'high_ambiguity': float(high_te_fc - high_te_cf),
                    'low_ambiguity': float(low_te_fc - low_te_cf),
                    'difference': float((high_te_fc - high_te_cf) - (low_te_fc - low_te_cf))
                }
                
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
                te_path = os.path.join(self.figure_dir, f'subject{subject_idx+1}_transfer_entropy.png')
                plt.savefig(te_path)
                plt.close()
                logger.info(f"  Saved transfer entropy analysis to {te_path}")
                
                # Save transfer entropy data
                te_data = {
                    'high_te_fc': float(high_te_fc),
                    'high_te_cf': float(high_te_cf),
                    'low_te_fc': float(low_te_fc),
                    'low_te_cf': float(low_te_cf),
                    'high_sig_fc': float(high_sig_fc),
                    'high_sig_cf': float(high_sig_cf),
                    'low_sig_fc': float(low_sig_fc),
                    'low_sig_cf': float(low_sig_cf)
                }
                np.save(os.path.join(self.stats_dir, f'subject{subject_idx+1}_te_data.npy'), te_data)
            
            except Exception as e:
                logger.warning(f"  Error in transfer entropy analysis: {e}")
                logger.warning(traceback.format_exc())
            
            # Save all quantum metrics to a JSON file
            metrics_file = os.path.join(self.stats_dir, f'subject{subject_idx+1}_quantum_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(quantum_metrics, f, indent=4)
            logger.info(f"  Saved quantum metrics to {metrics_file}")
            
            # Store in subjects_data
            if subject_idx in self.subjects_data:
                self.subjects_data[subject_idx]['quantum_metrics'] = quantum_metrics
                
            return quantum_metrics
                
        except Exception as e:
            logger.error(f"Error analyzing quantum signatures for subject {subject_idx+1}: {e}")
            logger.error(traceback.format_exc())
            return None
            
    def perform_group_analysis(self):
        """
        Perform group-level analysis across all processed subjects.
        
        This function aggregates data from individual subjects and performs
        statistical analyses to identify consistent patterns across subjects.
        
        Returns:
        --------
        dict
            Group-level metrics and statistics
        """
        logger.info("Performing group-level analysis...")
        
        if not self.subjects_data:
            logger.warning("No subject data available for group analysis")
            return None
        
        try:
            # Create group-level containers
            subject_indices = sorted(self.subjects_data.keys())
            logger.info(f"Analyzing data from {len(subject_indices)} subjects: {[i+1 for i in subject_indices]}")
            
            # 1. Frequency band power analysis
            band_powers_all = {}
            for subject_idx in subject_indices:
                if 'band_powers' in self.subjects_data[subject_idx]:
                    for band_name, band_data in self.subjects_data[subject_idx]['band_powers'].items():
                        if band_name not in band_powers_all:
                            band_powers_all[band_name] = {
                                'high_ambiguity': [],
                                'low_ambiguity': [],
                                'difference': []
                            }
                        
                        band_powers_all[band_name]['high_ambiguity'].append(band_data['high_ambiguity'])
                        band_powers_all[band_name]['low_ambiguity'].append(band_data['low_ambiguity'])
                        band_powers_all[band_name]['difference'].append(band_data['difference'])
            
            # Calculate statistics for band powers
            band_power_stats = {}
            for band_name, band_data in band_powers_all.items():
                band_power_stats[band_name] = {
                    'high_ambiguity': {
                        'mean': float(np.mean(band_data['high_ambiguity'])),
                        'std': float(np.std(band_data['high_ambiguity'])),
                        'median': float(np.median(band_data['high_ambiguity'])),
                        'min': float(np.min(band_data['high_ambiguity'])),
                        'max': float(np.max(band_data['high_ambiguity']))
                    },
                    'low_ambiguity': {
                        'mean': float(np.mean(band_data['low_ambiguity'])),
                        'std': float(np.std(band_data['low_ambiguity'])),
                        'median': float(np.median(band_data['low_ambiguity'])),
                        'min': float(np.min(band_data['low_ambiguity'])),
                        'max': float(np.max(band_data['low_ambiguity']))
                    },
                    'difference': {
                        'mean': float(np.mean(band_data['difference'])),
                        'std': float(np.std(band_data['difference'])),
                        'median': float(np.median(band_data['difference'])),
                        'min': float(np.min(band_data['difference'])),
                        'max': float(np.max(band_data['difference']))
                    }
                }
                
                # Perform t-test for significant differences
                if len(band_data['high_ambiguity']) > 1 and len(band_data['low_ambiguity']) > 1:
                    from scipy import stats
                    t_stat, p_val = stats.ttest_rel(band_data['high_ambiguity'], band_data['low_ambiguity'])
                    band_power_stats[band_name]['t_test'] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_val),
                        'significant': p_val < 0.05
                    }
            
            # 2. Quantum metrics aggregation
            quantum_metrics_all = {
                'multiscale_entropy': {
                    'high_ambiguity': [],
                    'low_ambiguity': [],
                    'difference': []
                },
                'phase_synchronization': {
                    'alpha': {
                        'high_ambiguity_psi': [],
                        'low_ambiguity_psi': [],
                        'psi_difference': []
                    },
                    'beta': {
                        'high_ambiguity_psi': [],
                        'low_ambiguity_psi': [],
                        'psi_difference': []
                    }
                },
                'transfer_entropy': {
                    'frontal_to_central': {
                        'high_ambiguity': [],
                        'low_ambiguity': [],
                        'difference': []
                    },
                    'central_to_frontal': {
                        'high_ambiguity': [],
                        'low_ambiguity': [],
                        'difference': []
                    },
                    'asymmetry': {
                        'high_ambiguity': [],
                        'low_ambiguity': [],
                        'difference': []
                    }
                }
            }
            
            # Collect quantum metrics from all subjects
            for subject_idx in subject_indices:
                if 'quantum_metrics' in self.subjects_data[subject_idx]:
                    metrics = self.subjects_data[subject_idx]['quantum_metrics']
                    
                    # Multiscale Entropy (use mean across scales for simplicity)
                    if 'multiscale_entropy' in metrics:
                        mse = metrics['multiscale_entropy']
                        if 'high_ambiguity' in mse and 'low_ambiguity' in mse:
                            # Calculate mean entropy across scales
                            high_mean = np.nanmean(mse['high_ambiguity'])
                            low_mean = np.nanmean(mse['low_ambiguity'])
                            diff_mean = high_mean - low_mean
                            
                            quantum_metrics_all['multiscale_entropy']['high_ambiguity'].append(high_mean)
                            quantum_metrics_all['multiscale_entropy']['low_ambiguity'].append(low_mean)
                            quantum_metrics_all['multiscale_entropy']['difference'].append(diff_mean)
                    
                    # Phase Synchronization
                    if 'phase_synchronization' in metrics:
                        ps = metrics['phase_synchronization']
                        if 'alpha' in ps:
                            quantum_metrics_all['phase_synchronization']['alpha']['high_ambiguity_psi'].append(
                                ps['alpha']['high_ambiguity_psi'])
                            quantum_metrics_all['phase_synchronization']['alpha']['low_ambiguity_psi'].append(
                                ps['alpha']['low_ambiguity_psi'])
                            quantum_metrics_all['phase_synchronization']['alpha']['psi_difference'].append(
                                ps['alpha']['psi_difference'])
                        
                        if 'beta' in ps:
                            quantum_metrics_all['phase_synchronization']['beta']['high_ambiguity_psi'].append(
                                ps['beta']['high_ambiguity_psi'])
                            quantum_metrics_all['phase_synchronization']['beta']['low_ambiguity_psi'].append(
                                ps['beta']['low_ambiguity_psi'])
                            quantum_metrics_all['phase_synchronization']['beta']['psi_difference'].append(
                                ps['beta']['psi_difference'])
                    
                    # Transfer Entropy
                    if 'transfer_entropy' in metrics:
                        te = metrics['transfer_entropy']
                        if 'frontal_to_central' in te:
                            quantum_metrics_all['transfer_entropy']['frontal_to_central']['high_ambiguity'].append(
                                te['frontal_to_central']['high_ambiguity'])
                            quantum_metrics_all['transfer_entropy']['frontal_to_central']['low_ambiguity'].append(
                                te['frontal_to_central']['low_ambiguity'])
                            quantum_metrics_all['transfer_entropy']['frontal_to_central']['difference'].append(
                                te['frontal_to_central']['difference'])
                        
                        if 'central_to_frontal' in te:
                            quantum_metrics_all['transfer_entropy']['central_to_frontal']['high_ambiguity'].append(
                                te['central_to_frontal']['high_ambiguity'])
                            quantum_metrics_all['transfer_entropy']['central_to_frontal']['low_ambiguity'].append(
                                te['central_to_frontal']['low_ambiguity'])
                            quantum_metrics_all['transfer_entropy']['central_to_frontal']['difference'].append(
                                te['central_to_frontal']['difference'])
                        
                        if 'asymmetry' in te:
                            quantum_metrics_all['transfer_entropy']['asymmetry']['high_ambiguity'].append(
                                te['asymmetry']['high_ambiguity'])
                            quantum_metrics_all['transfer_entropy']['asymmetry']['low_ambiguity'].append(
                                te['asymmetry']['low_ambiguity'])
                            quantum_metrics_all['transfer_entropy']['asymmetry']['difference'].append(
                                te['asymmetry']['difference'])
            
            # Calculate statistics for quantum metrics
            from scipy import stats
            quantum_stats = {}
            
            # Helper function for calculating stats
            def calculate_stats(data_array):
                if not data_array or len(data_array) < 1:
                    return None
                
                return {
                    'mean': float(np.nanmean(data_array)),
                    'std': float(np.nanstd(data_array)),
                    'median': float(np.nanmedian(data_array)),
                    'min': float(np.nanmin(data_array)),
                    'max': float(np.nanmax(data_array))
                }
            
            # Helper function for t-test
            def perform_ttest(array1, array2):
                if len(array1) > 1 and len(array2) > 1:
                    t_stat, p_val = stats.ttest_rel(array1, array2)
                    return {
                        't_statistic': float(t_stat),
                        'p_value': float(p_val),
                        'significant': p_val < 0.05
                    }
                return None
            
            # Process multiscale entropy
            mse_data = quantum_metrics_all['multiscale_entropy']
            quantum_stats['multiscale_entropy'] = {
                'high_ambiguity': calculate_stats(mse_data['high_ambiguity']),
                'low_ambiguity': calculate_stats(mse_data['low_ambiguity']),
                'difference': calculate_stats(mse_data['difference']),
                't_test': perform_ttest(mse_data['high_ambiguity'], mse_data['low_ambiguity'])
            }
            
            # Process phase synchronization
            for band in ['alpha', 'beta']:
                ps_data = quantum_metrics_all['phase_synchronization'][band]
                quantum_stats['phase_synchronization_' + band] = {
                    'high_ambiguity': calculate_stats(ps_data['high_ambiguity_psi']),
                    'low_ambiguity': calculate_stats(ps_data['low_ambiguity_psi']),
                    'difference': calculate_stats(ps_data['psi_difference']),
                    't_test': perform_ttest(ps_data['high_ambiguity_psi'], ps_data['low_ambiguity_psi'])
                }
            
            # Process transfer entropy
            for direction in ['frontal_to_central', 'central_to_frontal', 'asymmetry']:
                te_data = quantum_metrics_all['transfer_entropy'][direction]
                quantum_stats['transfer_entropy_' + direction] = {
                    'high_ambiguity': calculate_stats(te_data['high_ambiguity']),
                    'low_ambiguity': calculate_stats(te_data['low_ambiguity']),
                    'difference': calculate_stats(te_data['difference']),
                    't_test': perform_ttest(te_data['high_ambiguity'], te_data['low_ambiguity'])
                }
            
            # Create group results container
            group_results = {
                'n_subjects': len(subject_indices),
                'subject_ids': [i+1 for i in subject_indices],
                'band_power_stats': band_power_stats,
                'quantum_stats': quantum_stats,
                'raw_data': {
                    'band_powers': band_powers_all,
                    'quantum_metrics': quantum_metrics_all
                }
            }


           # Define a custom JSON encoder class to handle boolean values
           class CustomJSONEncoder(json.JSONEncoder):
               def default(self, obj):
                   if isinstance(obj, bool):
                       return int(obj)  # Convert boolean to integer (0 or 1)
                   elif isinstance(obj, np.ndarray):
                       return obj.tolist()  # Convert numpy arrays to lists
                   elif isinstance(obj, np.integer):
                       return int(obj)  # Convert numpy integers to Python integers
                   elif isinstance(obj, np.floating):
                       return float(obj)  # Convert numpy floats to Python floats
                   return super().default(obj)
            
            # Save group results
            group_file = os.path.join(self.group_dir, 'group_analysis_results.json')
            with open(group_file, 'w') as f:
                json.dump(group_results, f, indent=4, cls=CustomJSONEncoder)
            logger.info(f"Saved group analysis results to {group_file}")
            
            # Create summary visualizations
            
            # 1. Band power comparison across subjects
            plt.figure(figsize=(12, 8))
            band_names = list(band_power_stats.keys())
            x = np.arange(len(band_names))
            width = 0.35
            
            # Get mean values for each band
            high_means = [band_power_stats[band]['high_ambiguity']['mean'] for band in band_names]
            high_stds = [band_power_stats[band]['high_ambiguity']['std'] for band in band_names]
            low_means = [band_power_stats[band]['low_ambiguity']['mean'] for band in band_names]
            low_stds = [band_power_stats[band]['low_ambiguity']['std'] for band in band_names]
            
            # Create bar plot
            plt.bar(x - width/2, high_means, width, label='High Ambiguity', color='r', yerr=high_stds)
            plt.bar(x + width/2, low_means, width, label='Low Ambiguity', color='b', yerr=low_stds)
            
            # Add significance markers
            for i, band in enumerate(band_names):
                if 't_test' in band_power_stats[band] and band_power_stats[band]['t_test']['significant']:
                    plt.text(i, max(high_means[i], low_means[i]) + max(high_stds[i], low_stds[i]) + 0.5, 
                             '*', fontsize=14, ha='center')
            
            plt.xlabel('Frequency Band')
            plt.ylabel('Power (dB)')
            plt.title('Frequency Band Power Comparison Across Subjects')
            plt.xticks(x, band_names)
            plt.legend()
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            band_power_path = os.path.join(self.group_dir, 'group_band_power_comparison.png')
            plt.savefig(band_power_path)
            plt.close()
            logger.info(f"Saved group band power comparison to {band_power_path}")
            
            # 2. Quantum metrics comparison
            # MSE
            plt.figure(figsize=(10, 6))
            labels = ['Multiscale\nEntropy', 'Phase Sync\nAlpha', 'Phase Sync\nBeta', 
                      'TE\nF→C', 'TE\nC→F', 'TE\nAsymmetry']
            
            # Collect mean values for each metric
            high_means = [
                quantum_stats['multiscale_entropy']['high_ambiguity']['mean'],
                quantum_stats['phase_synchronization_alpha']['high_ambiguity']['mean'],
                quantum_stats['phase_synchronization_beta']['high_ambiguity']['mean'],
                quantum_stats['transfer_entropy_frontal_to_central']['high_ambiguity']['mean'],
                quantum_stats['transfer_entropy_central_to_frontal']['high_ambiguity']['mean'],
                quantum_stats['transfer_entropy_asymmetry']['high_ambiguity']['mean']
            ]
            
            low_means = [
                quantum_stats['multiscale_entropy']['low_ambiguity']['mean'],
                quantum_stats['phase_synchronization_alpha']['low_ambiguity']['mean'],
                quantum_stats['phase_synchronization_beta']['low_ambiguity']['mean'],
                quantum_stats['transfer_entropy_frontal_to_central']['low_ambiguity']['mean'],
                quantum_stats['transfer_entropy_central_to_frontal']['low_ambiguity']['mean'],
                quantum_stats['transfer_entropy_asymmetry']['low_ambiguity']['mean']
            ]
            
            # Calculate standard errors
            high_stds = [
                quantum_stats['multiscale_entropy']['high_ambiguity']['std'],
                quantum_stats['phase_synchronization_alpha']['high_ambiguity']['std'],
                quantum_stats['phase_synchronization_beta']['high_ambiguity']['std'],
                quantum_stats['transfer_entropy_frontal_to_central']['high_ambiguity']['std'],
                quantum_stats['transfer_entropy_central_to_frontal']['high_ambiguity']['std'],
                quantum_stats['transfer_entropy_asymmetry']['high_ambiguity']['std']
            ]
            
            low_stds = [
                quantum_stats['multiscale_entropy']['low_ambiguity']['std'],
                quantum_stats['phase_synchronization_alpha']['low_ambiguity']['std'],
                quantum_stats['phase_synchronization_beta']['low_ambiguity']['std'],
                quantum_stats['transfer_entropy_frontal_to_central']['low_ambiguity']['std'],
                quantum_stats['transfer_entropy_central_to_frontal']['low_ambiguity']['std'],
                quantum_stats['transfer_entropy_asymmetry']['low_ambiguity']['std']
            ]
            
            # Create plot
            x = np.arange(len(labels))
            width = 0.35
            
            plt.bar(x - width/2, high_means, width, label='High Ambiguity', color='r', yerr=high_stds)
            plt.bar(x + width/2, low_means, width, label='Low Ambiguity', color='b', yerr=low_stds)
            
            # Add significance markers
            t_test_results = [
                quantum_stats['multiscale_entropy']['t_test'],
                quantum_stats['phase_synchronization_alpha']['t_test'],
                quantum_stats['phase_synchronization_beta']['t_test'],
                quantum_stats['transfer_entropy_frontal_to_central']['t_test'],
                quantum_stats['transfer_entropy_central_to_frontal']['t_test'],
                quantum_stats['transfer_entropy_asymmetry']['t_test']
            ]
            
            for i, test in enumerate(t_test_results):
                if test and test['significant']:
                    plt.text(i, max(high_means[i], low_means[i]) + max(high_stds[i], low_stds[i]) + 0.02, 
                             '*', fontsize=14, ha='center')
            
            plt.xlabel('Quantum Metric')
            plt.ylabel('Mean Value')
            plt.title('Quantum Metrics Comparison Across Subjects')
            plt.xticks(x, labels)
            plt.legend()
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            quantum_metrics_path = os.path.join(self.group_dir, 'group_quantum_metrics_comparison.png')
            plt.savefig(quantum_metrics_path)
            plt.close()
            logger.info(f"Saved group quantum metrics comparison to {quantum_metrics_path}")
            
            # Store group results
            self.group_metrics = group_results
            
            return group_results
            
        except Exception as e:
            logger.error(f"Error performing group analysis: {e}")
            logger.error(traceback.format_exc())
            return None
            
    def run_single_subject_analysis(self, subject_idx):
        """Run complete analysis pipeline for a single subject."""
        logger.info(f"Running complete analysis for Subject {subject_idx+1}")
        
        # Process subject data
        subject_data = self.process_subject(subject_idx)
        if subject_data is None:
            logger.error(f"Failed to process data for Subject {subject_idx+1}")
            return False
        
        trials, ambiguity_values, button_values, rt_values, pres_moments, high_ambiguity_trials, med_ambiguity_trials, low_ambiguity_trials = subject_data
        
        # Compare high vs. low ambiguity
        comparison_results = self.compare_high_low_ambiguity(
            subject_idx, trials, high_ambiguity_trials, low_ambiguity_trials
        )
        
        if comparison_results is None:
            logger.warning(f"Could not compare high vs. low ambiguity for Subject {subject_idx+1}")
        
        # Analyze quantum signatures
        quantum_results = self.analyze_quantum_signatures(
            subject_idx, trials, high_ambiguity_trials, low_ambiguity_trials
        )
        
        if quantum_results is None:
            logger.warning(f"Could not analyze quantum signatures for Subject {subject_idx+1}")
        
        logger.info(f"Analysis for Subject {subject_idx+1} completed")
        return True
    
    def run_full_analysis(self, subject_range=None):
        """
        Run the complete analysis pipeline for all subjects.
        
        Parameters:
        -----------
        subject_range : tuple or None
            Range of subjects to analyze (start_idx, end_idx), if None analyze all subjects
        
        Returns:
        --------
        bool
            Success status
        """
        # Load data if not already loaded
        if self.eeg_data is None:
            if not self.load_data():
                return False
        
        # Determine subject range
        if subject_range is None:
            subject_indices = range(len(self.eeg_data))
        else:
            start_idx, end_idx = subject_range
            subject_indices = range(max(0, start_idx), min(len(self.eeg_data), end_idx + 1))
        
        logger.info(f"Starting analysis for subjects {list(subject_indices)}")
        
        # Process each subject
        success_count = 0
        for subject_idx in tqdm(subject_indices, desc="Processing Subjects"):
            if self.run_single_subject_analysis(subject_idx):
                success_count += 1
        
        logger.info(f"Successfully processed {success_count} out of {len(subject_indices)} subjects")
        
        # Perform group analysis if we have data from multiple subjects
        if success_count > 1:
            logger.info("Performing group-level analysis...")
            self.perform_group_analysis()
        
        return success_count > 0

def main():
    """Main function to parse arguments and run the analysis."""
    parser = argparse.ArgumentParser(description='Analyze Necker Cube EEG dataset')
    parser.add_argument('--data', type=str, default=None, help='Path to the EEG data file')
    parser.add_argument('--output', type=str, default=None, help='Output directory for results')
    parser.add_argument('--subject', type=int, default=None, help='Single subject to analyze (1-based index)')
    parser.add_argument('--start', type=int, default=0, help='First subject index to analyze (0-based)')
    parser.add_argument('--end', type=int, default=None, help='Last subject index to analyze (0-based)')
    parser.add_argument('--skip-group', action='store_true', help='Skip group-level analysis')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = DatasetProcessor(data_path=args.data, output_base_dir=args.output)
    
    # Load data
    if not processor.load_data():
        logger.error("Failed to load data, exiting")
        return 1
    
    # Run analysis based on arguments
    if args.subject is not None:
        # Analyze a single subject (convert from 1-based to 0-based indexing)
        subject_idx = args.subject - 1
        if subject_idx < 0 or subject_idx >= len(processor.eeg_data):
            logger.error(f"Subject index {args.subject} out of range (1-{len(processor.eeg_data)})")
            return 1
        
        success = processor.run_single_subject_analysis(subject_idx)
        
        # Run group analysis if requested
        if success and not args.skip_group:
            processor.perform_group_analysis()
    else:
        # Analyze a range of subjects
        end_idx = args.end if args.end is not None else len(processor.eeg_data) - 1
        subject_range = (args.start, end_idx)
        
        success = processor.run_full_analysis(subject_range)
        
        if not success:
            logger.error("Analysis failed")
            return 1
    
    logger.info("Analysis completed successfully")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
