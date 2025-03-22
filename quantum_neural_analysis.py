#!/usr/bin/env python
"""
Quantum Neural Coherence Analysis Script
This script analyzes EEG data from the Necker cube experiment, focusing on quantum signatures
in neural activity during perceptual decision-making under varying ambiguity conditions.

Author: Claude Analysis
Date: March 22, 2025
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec

# Set paths
project_dir = '/blue/mcb4324/share/sdavidson2/QuantumNeuralCoherence'
results_dir = os.path.join(project_dir, 'results/necker_cube_eeg_Maksimenko')
stats_dir = os.path.join(results_dir, 'statistics')
figures_dir = os.path.join(results_dir, 'figures')
group_dir = os.path.join(results_dir, 'group_analysis')
analysis_dir = os.path.join(results_dir, 'analysis')

# Create analysis directory if it doesn't exist
os.makedirs(analysis_dir, exist_ok=True)

# Set up plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("paper", font_scale=1.5)
colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

# Helper functions
def load_json_data(file_path):
    """Load data from a JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from: {file_path}")
        return None

def safe_mean(data_list):
    """Calculate mean of a list, handling empty lists and non-numeric values"""
    filtered_data = [x for x in data_list if isinstance(x, (int, float))]
    if filtered_data:
        return np.mean(filtered_data)
    return np.nan

def safe_std(data_list):
    """Calculate standard deviation of a list, handling empty lists and non-numeric values"""
    filtered_data = [x for x in data_list if isinstance(x, (int, float))]
    if len(filtered_data) > 1:
        return np.std(filtered_data)
    return np.nan

def calculate_effect_size(group1, group2):
    """Calculate Cohen's d effect size between two groups"""
    filtered1 = [x for x in group1 if isinstance(x, (int, float))]
    filtered2 = [x for x in group2 if isinstance(x, (int, float))]
    
    if not filtered1 or not filtered2:
        return np.nan
    
    mean1, mean2 = np.mean(filtered1), np.mean(filtered2)
    var1, var2 = np.var(filtered1, ddof=1), np.var(filtered2, ddof=1)
    n1, n2 = len(filtered1), len(filtered2)
    
    # Pooled standard deviation
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_sd == 0:
        return np.nan
    
    return (mean1 - mean2) / pooled_sd

def paired_ttest(group1, group2, alpha=0.05):
    """Perform paired t-test between two groups"""
    # Filter to ensure we only have numeric values
    paired_data = [(x, y) for x, y in zip(group1, group2) 
                   if isinstance(x, (int, float)) and isinstance(y, (int, float))]
    
    if not paired_data:
        return {'t_stat': np.nan, 'p_value': np.nan, 'significant': False}
    
    x_values, y_values = zip(*paired_data)
    t_stat, p_value = stats.ttest_rel(x_values, y_values)
    
    return {
        't_stat': t_stat,
        'p_value': p_value,
        'significant': p_value < alpha
    }

print("Starting Quantum Neural Coherence Analysis...")

# ------------------------
# 1. Collect and Organize Data
# ------------------------
print("Collecting data across subjects...")

n_subjects = 20
subject_data = {}

# Collect band powers across subjects
all_band_powers = {}
for subject_idx in range(1, n_subjects + 1):
    subject_file = os.path.join(stats_dir, f'subject{subject_idx}_band_powers.json')
    if os.path.exists(subject_file):
        with open(subject_file, 'r') as f:
            try:
                band_data = json.load(f)
                all_band_powers[subject_idx] = band_data
                
                # Add to overall subject data
                if subject_idx not in subject_data:
                    subject_data[subject_idx] = {}
                subject_data[subject_idx]['band_powers'] = band_data
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {subject_file}")

# Collect quantum metrics across subjects
all_quantum_metrics = {}
for subject_idx in range(1, n_subjects + 1):
    metrics_file = os.path.join(stats_dir, f'subject{subject_idx}_quantum_metrics.json')
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            try:
                metrics_data = json.load(f)
                all_quantum_metrics[subject_idx] = metrics_data
                
                # Add to overall subject data
                if subject_idx not in subject_data:
                    subject_data[subject_idx] = {}
                subject_data[subject_idx]['quantum_metrics'] = metrics_data
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {metrics_file}")

# Check if we have data
if not all_band_powers and not all_quantum_metrics:
    print("No data found! Make sure the file paths are correct.")
    exit(1)

print(f"Collected data for {len(subject_data)} subjects")

# ------------------------
# 2. Frequency Band Power Analysis
# ------------------------
print("Analyzing frequency band powers...")

# Collect frequency band data for all subjects
band_power_df = []
for subject_idx, subject_band_data in all_band_powers.items():
    for band_name, band_values in subject_band_data.items():
        band_power_df.append({
            'subject': subject_idx,
            'band': band_name,
            'high_ambiguity': band_values.get('high_ambiguity', np.nan),
            'low_ambiguity': band_values.get('low_ambiguity', np.nan),
            'difference': band_values.get('difference', np.nan)
        })

# Convert to DataFrame for easier analysis
band_df = pd.DataFrame(band_power_df)

# Create a summary of the band power differences
band_summary = band_df.groupby('band').agg({
    'high_ambiguity': ['mean', 'std', 'count'],
    'low_ambiguity': ['mean', 'std', 'count'],
    'difference': ['mean', 'std', 'count']
})

# Perform paired t-tests for each band
band_stats = []
for band_name in band_df['band'].unique():
    band_data = band_df[band_df['band'] == band_name]
    high_values = band_data['high_ambiguity'].tolist()
    low_values = band_data['low_ambiguity'].tolist()
    
    t_test_result = paired_ttest(high_values, low_values)
    effect_size = calculate_effect_size(high_values, low_values)
    
    band_stats.append({
        'band': band_name,
        'high_mean': safe_mean(high_values),
        'high_std': safe_std(high_values),
        'low_mean': safe_mean(low_values),
        'low_std': safe_std(low_values),
        'diff_mean': safe_mean(band_data['difference'].tolist()),
        'diff_std': safe_std(band_data['difference'].tolist()),
        't_stat': t_test_result['t_stat'],
        'p_value': t_test_result['p_value'],
        'significant': t_test_result['significant'],
        'effect_size': effect_size,
        'n_subjects': len(band_data)
    })

band_stats_df = pd.DataFrame(band_stats)

# Plot frequency band differences
plt.figure(figsize=(12, 6))
sns.barplot(x='band', y='difference', data=band_df, errorbar=('ci', 95), palette='viridis')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.title('Difference in Band Power (High - Low Ambiguity)', fontsize=16)
plt.ylabel('Power Difference (dB)', fontsize=14)
plt.xlabel('Frequency Band', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add significance markers
for i, band in enumerate(band_stats_df['band']):
    band_row = band_stats_df[band_stats_df['band'] == band].iloc[0]
    if band_row['significant']:
        plt.text(i, band_row['diff_mean'] + 0.1, '*', 
                 ha='center', va='center', color='black', fontsize=20)

plt.tight_layout()
plt.savefig(os.path.join(analysis_dir, 'band_power_differences.png'), dpi=300)

# ------------------------
# 3. Quantum Metrics Analysis
# ------------------------
print("Analyzing quantum neural signatures...")

# Organize MSE data
mse_data = []
for subject_idx, metrics in all_quantum_metrics.items():
    if 'multiscale_entropy' in metrics:
        mse = metrics['multiscale_entropy']
        high_values = mse.get('high_ambiguity', [])
        low_values = mse.get('low_ambiguity', [])
        diff_values = mse.get('difference', [])
        
        # Make sure we have lists
        if not isinstance(high_values, list):
            high_values = [high_values]
        if not isinstance(low_values, list):
            low_values = [low_values]
        if not isinstance(diff_values, list):
            diff_values = [diff_values]
        
        # Calculate means if we have values
        high_mean = safe_mean(high_values)
        low_mean = safe_mean(low_values)
        diff_mean = safe_mean(diff_values)
        
        mse_data.append({
            'subject': subject_idx,
            'metric': 'Multiscale Entropy',
            'high_ambiguity': high_mean,
            'low_ambiguity': low_mean,
            'difference': diff_mean
        })

# Organize Phase Synchronization data
ps_data = []
for subject_idx, metrics in all_quantum_metrics.items():
    if 'phase_synchronization' in metrics:
        ps = metrics['phase_synchronization']
        
        # Process alpha band
        if 'alpha' in ps:
            alpha_ps = ps['alpha']
            ps_data.append({
                'subject': subject_idx,
                'metric': 'Phase Sync (Alpha)',
                'high_ambiguity': alpha_ps.get('high_ambiguity_psi', np.nan),
                'low_ambiguity': alpha_ps.get('low_ambiguity_psi', np.nan),
                'difference': alpha_ps.get('psi_difference', np.nan)
            })
        
        # Process beta band
        if 'beta' in ps:
            beta_ps = ps['beta']
            ps_data.append({
                'subject': subject_idx,
                'metric': 'Phase Sync (Beta)',
                'high_ambiguity': beta_ps.get('high_ambiguity_psi', np.nan),
                'low_ambiguity': beta_ps.get('low_ambiguity_psi', np.nan),
                'difference': beta_ps.get('psi_difference', np.nan)
            })

# Organize Transfer Entropy data
te_data = []
for subject_idx, metrics in all_quantum_metrics.items():
    if 'transfer_entropy' in metrics:
        te = metrics['transfer_entropy']
        
        # Process F→C direction
        if 'frontal_to_central' in te:
            fc_te = te['frontal_to_central']
            te_data.append({
                'subject': subject_idx,
                'metric': 'TE (F→C)',
                'high_ambiguity': fc_te.get('high_ambiguity', np.nan),
                'low_ambiguity': fc_te.get('low_ambiguity', np.nan),
                'difference': fc_te.get('difference', np.nan)
            })
        
        # Process C→F direction
        if 'central_to_frontal' in te:
            cf_te = te['central_to_frontal']
            te_data.append({
                'subject': subject_idx,
                'metric': 'TE (C→F)',
                'high_ambiguity': cf_te.get('high_ambiguity', np.nan),
                'low_ambiguity': cf_te.get('low_ambiguity', np.nan),
                'difference': cf_te.get('difference', np.nan)
            })
        
        # Process asymmetry
        if 'asymmetry' in te:
            asym_te = te['asymmetry']
            te_data.append({
                'subject': subject_idx,
                'metric': 'TE Asymmetry',
                'high_ambiguity': asym_te.get('high_ambiguity', np.nan),
                'low_ambiguity': asym_te.get('low_ambiguity', np.nan),
                'difference': asym_te.get('difference', np.nan)
            })

# Combine all quantum metrics
quantum_df = pd.concat([
    pd.DataFrame(mse_data),
    pd.DataFrame(ps_data),
    pd.DataFrame(te_data)
])

# Calculate statistics for each quantum metric
quantum_stats = []
for metric_name in quantum_df['metric'].unique():
    metric_data = quantum_df[quantum_df['metric'] == metric_name]
    high_values = metric_data['high_ambiguity'].tolist()
    low_values = metric_data['low_ambiguity'].tolist()
    
    t_test_result = paired_ttest(high_values, low_values)
    effect_size = calculate_effect_size(high_values, low_values)
    
    quantum_stats.append({
        'metric': metric_name,
        'high_mean': safe_mean(high_values),
        'high_std': safe_std(high_values),
        'low_mean': safe_mean(low_values),
        'low_std': safe_std(low_values),
        'diff_mean': safe_mean(metric_data['difference'].tolist()),
        'diff_std': safe_std(metric_data['difference'].tolist()),
        't_stat': t_test_result['t_stat'],
        'p_value': t_test_result['p_value'],
        'significant': t_test_result['significant'],
        'effect_size': effect_size,
        'n_subjects': len(metric_data)
    })

quantum_stats_df = pd.DataFrame(quantum_stats)

# Plot quantum metrics differences
plt.figure(figsize=(15, 7))
sns.barplot(x='metric', y='difference', data=quantum_df, errorbar=('ci', 95), palette='coolwarm')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.title('Quantum Metrics Difference (High - Low Ambiguity)', fontsize=16)
plt.ylabel('Difference', fontsize=14)
plt.xlabel('Metric', fontsize=14)
plt.xticks(fontsize=12, rotation=30, ha='right')
plt.yticks(fontsize=12)

# Add significance markers
for i, metric in enumerate(quantum_stats_df['metric']):
    metric_row = quantum_stats_df[quantum_stats_df['metric'] == metric].iloc[0]
    if metric_row['significant']:
        y_pos = metric_row['diff_mean'] + 0.01 if metric_row['diff_mean'] > 0 else metric_row['diff_mean'] - 0.01
        plt.text(i, y_pos, '*', 
                 ha='center', va='center', color='black', fontsize=20)

plt.tight_layout()
plt.savefig(os.path.join(analysis_dir, 'quantum_metrics_differences.png'), dpi=300)

# ------------------------
# 4. Correlation Analysis
# ------------------------
print("Performing correlation analysis...")

# Create a wide format DataFrame for correlations
quantum_wide = quantum_df.pivot_table(
    index='subject', 
    columns='metric', 
    values='difference'
)

# Add band power differences
band_wide = band_df.pivot_table(
    index='subject', 
    columns='band', 
    values='difference',
    aggfunc='first'  # In case of duplicates
)

# Combine the datasets
combined_wide = pd.concat([quantum_wide, band_wide], axis=1)

# Calculate correlation matrix
corr_matrix = combined_wide.corr()

# Plot correlation heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
            linewidths=0.5, annot_kws={"size": 8})
plt.title('Correlation Between Quantum Metrics and Band Powers', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(analysis_dir, 'correlation_matrix.png'), dpi=300)

# ------------------------
# 5. Identify Key Patterns
# ------------------------
print("Identifying key patterns...")

# Look for significant metrics
significant_metrics = quantum_stats_df[quantum_stats_df['significant']].sort_values(by='effect_size', ascending=False)
significant_bands = band_stats_df[band_stats_df['significant']].sort_values(by='effect_size', ascending=False)

# Create a summary of key findings
with open(os.path.join(analysis_dir, 'key_findings.txt'), 'w') as f:
    f.write("QUANTUM NEURAL COHERENCE ANALYSIS - KEY FINDINGS\n")
    f.write("==============================================\n\n")
    
    f.write("1. SIGNIFICANT FREQUENCY BAND DIFFERENCES:\n")
    if len(significant_bands) > 0:
        for _, band in significant_bands.iterrows():
            f.write(f"   * {band['band']} Band: High Ambiguity ({band['high_mean']:.3f} ± {band['high_std']:.3f}) vs. ")
            f.write(f"Low Ambiguity ({band['low_mean']:.3f} ± {band['low_std']:.3f})\n")
            f.write(f"     Difference: {band['diff_mean']:.3f} ± {band['diff_std']:.3f}, ")
            f.write(f"t({band['n_subjects']-1}) = {band['t_stat']:.3f}, p = {band['p_value']:.4f}, ")
            f.write(f"Cohen's d = {band['effect_size']:.3f}\n")
    else:
        f.write("   * No significant frequency band differences found.\n")
    
    f.write("\n2. SIGNIFICANT QUANTUM METRICS:\n")
    if len(significant_metrics) > 0:
        for _, metric in significant_metrics.iterrows():
            f.write(f"   * {metric['metric']}: High Ambiguity ({metric['high_mean']:.3f} ± {metric['high_std']:.3f}) vs. ")
            f.write(f"Low Ambiguity ({metric['low_mean']:.3f} ± {metric['low_std']:.3f})\n")
            f.write(f"     Difference: {metric['diff_mean']:.3f} ± {metric['diff_std']:.3f}, ")
            f.write(f"t({metric['n_subjects']-1}) = {metric['t_stat']:.3f}, p = {metric['p_value']:.4f}, ")
            f.write(f"Cohen's d = {metric['effect_size']:.3f}\n")
    else:
        f.write("   * No significant quantum metrics found.\n")
    
    f.write("\n3. STRONG CORRELATIONS (|r| > 0.5):\n")
    strong_correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                strong_correlations.append((col1, col2, corr_val))
    
    if strong_correlations:
        for col1, col2, corr_val in sorted(strong_correlations, key=lambda x: abs(x[2]), reverse=True):
            f.write(f"   * {col1} and {col2}: r = {corr_val:.3f}\n")
    else:
        f.write("   * No strong correlations found.\n")
    
    f.write("\n4. INTERPRETATION AND QUANTUM SIGNATURES:\n")
    
    # Check for patterns that suggest quantum signatures
    mse_metrics = quantum_stats_df[quantum_stats_df['metric'] == 'Multiscale Entropy']
    phase_sync_alpha = quantum_stats_df[quantum_stats_df['metric'] == 'Phase Sync (Alpha)']
    phase_sync_beta = quantum_stats_df[quantum_stats_df['metric'] == 'Phase Sync (Beta)']
    te_asymmetry = quantum_stats_df[quantum_stats_df['metric'] == 'TE Asymmetry']
    
    # MSE interpretation
    if not mse_metrics.empty:
        mse_diff = mse_metrics.iloc[0]['diff_mean']
        if mse_metrics.iloc[0]['significant']:
            if mse_diff > 0:
                f.write("   * Multiscale Entropy is significantly HIGHER during high ambiguity, suggesting\n")
                f.write("     increased neural complexity and potential quantum superposition-like states\n")
                f.write("     during perceptual uncertainty.\n")
            else:
                f.write("   * Multiscale Entropy is significantly LOWER during high ambiguity, suggesting\n")
                f.write("     reduced neural complexity during perceptual uncertainty.\n")
    
    # Phase synchronization interpretation
    if not phase_sync_alpha.empty and phase_sync_alpha.iloc[0]['significant']:
        ps_alpha_diff = phase_sync_alpha.iloc[0]['diff_mean']
        if ps_alpha_diff > 0:
            f.write("   * Alpha band Phase Synchronization is significantly HIGHER during high ambiguity,\n")
            f.write("     suggesting quantum entanglement-like effects in alpha oscillations during\n")
            f.write("     perceptual uncertainty.\n")
        else:
            f.write("   * Alpha band Phase Synchronization is significantly LOWER during high ambiguity,\n")
            f.write("     suggesting less coherent alpha oscillations during perceptual uncertainty.\n")
    
    if not phase_sync_beta.empty and phase_sync_beta.iloc[0]['significant']:
        ps_beta_diff = phase_sync_beta.iloc[0]['diff_mean']
        if ps_beta_diff > 0:
            f.write("   * Beta band Phase Synchronization is significantly HIGHER during high ambiguity,\n")
            f.write("     suggesting quantum entanglement-like effects in beta oscillations during\n")
            f.write("     perceptual uncertainty.\n")
        else:
            f.write("   * Beta band Phase Synchronization is significantly LOWER during high ambiguity,\n")
            f.write("     suggesting less coherent beta oscillations during perceptual uncertainty.\n")
    
    # Transfer entropy interpretation
    if not te_asymmetry.empty and te_asymmetry.iloc[0]['significant']:
        te_asym_diff = te_asymmetry.iloc[0]['diff_mean']
        if te_asym_diff > 0:
            f.write("   * Transfer Entropy Asymmetry is significantly HIGHER during high ambiguity,\n")
            f.write("     suggesting more directed information flow and potential quantum non-locality\n")
            f.write("     during perceptual uncertainty.\n")
        else:
            f.write("   * Transfer Entropy Asymmetry is significantly LOWER during high ambiguity,\n")
            f.write("     suggesting less directed information flow during perceptual uncertainty.\n")
    
    # Check for alpha-beta band differences
    alpha_band = band_stats_df[band_stats_df['band'] == 'alpha']
    beta_band = band_stats_df[band_stats_df['band'] == 'beta']
    
    if not alpha_band.empty and alpha_band.iloc[0]['significant']:
        alpha_diff = alpha_band.iloc[0]['diff_mean']
        if alpha_diff > 0:
            f.write("   * Alpha band power is significantly HIGHER during high ambiguity, suggesting\n")
            f.write("     increased inhibitory processing during perceptual uncertainty.\n")
        else:
            f.write("   * Alpha band power is significantly LOWER during high ambiguity, suggesting\n")
            f.write("     decreased inhibitory processing during perceptual uncertainty.\n")
    
    if not beta_band.empty and beta_band.iloc[0]['significant']:
        beta_diff = beta_band.iloc[0]['diff_mean']
        if beta_diff > 0:
            f.write("   * Beta band power is significantly HIGHER during high ambiguity, suggesting\n")
            f.write("     increased active processing during perceptual uncertainty.\n")
        else:
            f.write("   * Beta band power is significantly LOWER during high ambiguity, suggesting\n")
            f.write("     decreased active processing during perceptual uncertainty.\n")
    
    # Overall quantum interpretation
    f.write("\n5. QUANTUM VS. CLASSICAL INTERPRETATION:\n")
    
    # Count significant quantum metrics
    sig_quantum_count = len(significant_metrics)
    
    if sig_quantum_count >= 2:
        f.write("   The analysis reveals multiple significant quantum metrics, suggesting\n")
        f.write("   potential quantum signatures in neural processing during perceptual\n")
        f.write("   ambiguity. The combination of:\n")
        
        for _, metric in significant_metrics.iterrows():
            diff = "higher" if metric['diff_mean'] > 0 else "lower"
            f.write(f"   - {diff} {metric['metric']}\n")
        
        f.write("\n   This pattern is consistent with quantum processing theories including:\n")
        f.write("   - Superposition-like states during perceptual uncertainty\n")
        f.write("   - Entanglement-like neural synchronization\n")
        f.write("   - Non-local information transfer\n")
        
        f.write("\n   However, classical interpretations remain viable, including:\n")
        f.write("   - Increased processing demands during ambiguity\n")
        f.write("   - Enhanced attentional allocation\n")
        f.write("   - Greater coordination between brain regions\n")
    elif sig_quantum_count == 1:
        f.write("   The analysis reveals one significant quantum metric, which provides\n")
        f.write("   limited evidence for quantum signatures in neural processing.\n")
        f.write("   While intriguing, this finding should be interpreted cautiously\n")
        f.write("   as it could also be explained by classical neural mechanisms.\n")
    else:
        f.write("   The analysis does not reveal significant quantum metrics,\n")
        f.write("   suggesting that the neural processing during this perceptual\n")
        f.write("   task is adequately explained by classical mechanisms.\n")

# ------------------------
# 6. Create Summary Visualization
# ------------------------
print("Creating summary visualization...")

# Create a comprehensive figure showing key results
plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

# 1. Band Power Differences
ax1 = plt.subplot(gs[0, 0])
sns.barplot(x='band', y='difference', data=band_df, errorbar=('ci', 95), palette='viridis', ax=ax1)
ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax1.set_title('Frequency Band Power Differences\n(High - Low Ambiguity)', fontsize=14)
ax1.set_ylabel('Power Difference (dB)', fontsize=12)
ax1.set_xlabel('Frequency Band', fontsize=12)

# Add significance markers
for i, band in enumerate(band_stats_df['band']):
    band_row = band_stats_df[band_stats_df['band'] == band].iloc[0]
    if band_row['significant']:
        ax1.text(i, band_row['diff_mean'] + 0.1, '*', 
                 ha='center', va='center', color='black', fontsize=20)

# 2. Quantum Metrics Differences
ax2 = plt.subplot(gs[0, 1])
sns.barplot(x='metric', y='difference', data=quantum_df, errorbar=('ci', 95), palette='coolwarm', ax=ax2)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax2.set_title('Quantum Metrics Differences\n(High - Low Ambiguity)', fontsize=14)
ax2.set_ylabel('Difference', fontsize=12)
ax2.set_xlabel('Metric', fontsize=12)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

# Add significance markers
for i, metric in enumerate(quantum_stats_df['metric']):
    metric_row = quantum_stats_df[quantum_stats_df['metric'] == metric].iloc[0]
    if metric_row['significant']:
        y_pos = metric_row['diff_mean'] + 0.01 if metric_row['diff_mean'] > 0 else metric_row['diff_mean'] - 0.01
        ax2.text(i, y_pos, '*', 
                 ha='center', va='center', color='black', fontsize=20)

# 3. Correlation Heatmap (Simplified)
ax3 = plt.subplot(gs[1, :])
# Select only important metrics for cleaner visualization
key_metrics = ['Multiscale Entropy', 'Phase Sync (Alpha)', 'Phase Sync (Beta)', 
               'TE Asymmetry', 'alpha', 'beta', 'gamma', 'delta', 'theta']
key_metrics = [m for m in key_metrics if m in corr_matrix.columns]  # Only use metrics that exist
key_corr = corr_matrix.loc[key_metrics, key_metrics]
sns.heatmap(key_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
            linewidths=0.5, annot_kws={"size": 10}, ax=ax3)
ax3.set_title('Correlations Between Key Metrics', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(analysis_dir, 'quantum_neural_summary.png'), dpi=300)

# ------------------------
# 7. Generate Individual Subject Profiles
# ------------------------
print("Generating individual subject profiles...")

# Identify subjects with strongest quantum signatures
subject_quantum_scores = {}
for subject in quantum_df['subject'].unique():
    subject_data = quantum_df[quantum_df['subject'] == subject]
    
    # Calculate a "quantum signature score" based on difference magnitudes in key metrics
    # Higher = stronger quantum-like patterns
    quantum_score = 0
    
    # Check MSE
    mse_row = subject_data[subject_data['metric'] == 'Multiscale Entropy']
    if not mse_row.empty and not np.isnan(mse_row['difference'].values[0]):
        # Higher MSE in high ambiguity condition is a positive quantum signature
        quantum_score += mse_row['difference'].values[0]
    
    # Check Phase Sync Alpha
    ps_alpha_row = subject_data[subject_data['metric'] == 'Phase Sync (Alpha)']
    if not ps_alpha_row.empty and not np.isnan(ps_alpha_row['difference'].values[0]):
        # Higher phase sync in high ambiguity condition is a positive quantum signature
        quantum_score += ps_alpha_row['difference'].values[0]
    
    # Check TE Asymmetry
    te_asym_row = subject_data[subject_data['metric'] == 'TE Asymmetry']
    if not te_asym_row.empty and not np.isnan(te_asym_row['difference'].values[0]):
        # Higher TE asymmetry in high ambiguity condition is a positive quantum signature
        quantum_score += te_asym_row['difference'].values[0]
    
    subject_quantum_scores[subject] = quantum_score

# Sort subjects by quantum signature strength
sorted_subjects = sorted(subject_quantum_scores.items(), key=lambda x: abs(x[1]), reverse=True)

# Create a report of top 5 subjects with strongest quantum signatures
with open(os.path.join(analysis_dir, 'top_subjects.txt'), 'w') as f:
    f.write("SUBJECTS WITH STRONGEST QUANTUM SIGNATURES\n")
    f.write("=========================================\n\n")
    
    for subject, score in sorted_subjects[:5]:
        f.write(f"Subject {subject} (Quantum Score: {score:.3f})\n")
        f.write("-" * 50 + "\n")
        
        # Write band power info
        subject_band_data = band_df[band_df['subject'] == subject]
        f.write("Frequency Bands:\n")
        for _, band_row in subject_band_data.iterrows():
            band_name = band_row['band']
            high_val = band_row['high_ambiguity']
            low_val = band_row['low_ambiguity']
            diff_val = band_row['difference']
            f.write(f"  * {band_name}: High={high_val:.3f}, Low={low_val:.3f}, Diff={diff_val:.3f}\n")
        
        # Write quantum metrics info
        subject_quantum_data = quantum_df[quantum_df['subject'] == subject]
        f.write("\nQuantum Metrics:\n")
        for _, metric_row in subject_quantum_data.iterrows():
            metric_name = metric_row['metric']
            high_val = metric_row['high_ambiguity']
            low_val = metric_row['low_ambiguity']
            diff_val = metric_row['difference']
            f.write(f"  * {metric_name}: High={high_val:.3f}, Low={low_val:.3f}, Diff={diff_val:.3f}\n")
        
        f.write("\n\n")

# ------------------------
# 8. Final Summary Report
# ------------------------
print("Generating final summary report...")

with open(os.path.join(analysis_dir, 'quantum_neural_report.md'), 'w') as f:
    f.write("# Quantum Neural Coherence Analysis Report\n\n")
    f.write("## Analysis of Necker Cube EEG Dataset\n")
    f.write(f"*Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d')}*\n\n")
    
    f.write("## Overview\n\n")
    f.write("This report analyzes EEG data from the Necker Cube experiment, focusing on quantum neural ")
    f.write("signatures during perceptual decision-making under varying ambiguity conditions. ")
    f.write("The analysis compares neural activity between high and low ambiguity conditions to identify ")
    f.write("potential quantum signatures in brain function.\n\n")
    
    f.write("## Key Findings\n\n")
    
    # Significant Frequency Bands
    f.write("### Frequency Band Differences\n\n")
    f.write("| Band | High Ambiguity | Low Ambiguity | Difference | p-value | Significant |\n")
    f.write("|------|---------------|---------------|------------|---------|-------------|\n")
    for _, band in band_stats_df.iterrows():
        sig_mark = "✓" if band['significant'] else "✗"
        f.write(f"| {band['band']} | {band['high_mean']:.3f} ± {band['high_std']:.3f} | ")
        f.write(f"{band['low_mean']:.3f} ± {band['low_std']:.3f} | ")
        f.write(f"{band['diff_mean']:.3f} ± {band['diff_std']:.3f} | ")
        f.write(f"{band['p_value']:.4f} | {sig_mark} |\n")
    
    f.write("\n")
    
    # Significant Quantum Metrics
    f.write("### Quantum Neural Metrics\n\n")
    f.write("| Metric | High Ambiguity | Low Ambiguity | Difference | p-value | Significant |\n")
    f.write("|--------|---------------|---------------|------------|---------|-------------|\n")
    for _, metric in quantum_stats_df.iterrows():
        sig_mark = "✓" if metric['significant'] else "✗"
        f.write(f"| {metric['metric']} | {metric['high_mean']:.3f} ± {metric['high_std']:.3f} | ")
        f.write(f"{metric['low_mean']:.3f} ± {metric['low_std']:.3f} | ")
        f.write(f"{metric['diff_mean']:.3f} ± {metric['diff_std']:.3f} | ")
        f.write(f"{metric['p_value']:.4f} | {sig_mark} |\n")
    
    f.write("\n")
    
    f.write("### Quantum Signatures Summary\n\n")
    
    # Count significant metrics with positive differences (high > low ambiguity)
    positive_sig_metrics = quantum_stats_df[(quantum_stats_df['significant']) & 
                                              (quantum_stats_df['diff_mean'] > 0)]
    negative_sig_metrics = quantum_stats_df[(quantum_stats_df['significant']) & 
                                              (quantum_stats_df['diff_mean'] < 0)]
    
    if len(positive_sig_metrics) > 0:
        f.write("#### Quantum Signatures (High > Low Ambiguity):\n\n")
        for _, metric in positive_sig_metrics.iterrows():
            f.write(f"- **{metric['metric']}**: Higher during perceptual uncertainty ")
            f.write(f"(diff: {metric['diff_mean']:.3f}, p={metric['p_value']:.4f})\n")
        f.write("\n")
    
    if len(negative_sig_metrics) > 0:
        f.write("#### Anti-Quantum Signatures (Low > High Ambiguity):\n\n")
        for _, metric in negative_sig_metrics.iterrows():
            f.write(f"- **{metric['metric']}**: Lower during perceptual uncertainty ")
            f.write(f"(diff: {metric['diff_mean']:.3f}, p={metric['p_value']:.4f})\n")
        f.write("\n")
    
    # Overall interpretation
    f.write("### Interpretation\n\n")
    
    if len(significant_metrics) >= 2:
        f.write("The analysis reveals **multiple significant quantum metrics**, suggesting ")
        f.write("potential quantum signatures in neural processing during perceptual ")
        f.write("ambiguity. The pattern of results is consistent with quantum processing theories including:\n\n")
        
        f.write("- **Superposition-like states** during perceptual uncertainty\n")
        f.write("- **Entanglement-like neural synchronization** between brain regions\n")
        f.write("- **Non-local information transfer** during ambiguous perception\n\n")
        
        f.write("However, classical interpretations remain viable, including increased processing demands ")
        f.write("during ambiguity, enhanced attentional allocation, and greater coordination between brain regions.\n\n")
    elif len(significant_metrics) == 1:
        f.write("The analysis reveals **one significant quantum metric**, which provides ")
        f.write("limited evidence for quantum signatures in neural processing. ")
        f.write("While intriguing, this finding should be interpreted cautiously ")
        f.write("as it could also be explained by classical neural mechanisms.\n\n")
    else:
        f.write("The analysis does not reveal significant quantum metrics, ")
        f.write("suggesting that the neural processing during this perceptual ")
        f.write("task is adequately explained by classical mechanisms.\n\n")
    
    # Subject variability
    f.write("### Individual Differences\n\n")
    f.write("There is considerable variation between subjects in quantum neural signatures. ")
    f.write(f"The 5 subjects with the strongest quantum signatures (Subjects ")
    top_5_subjects = [str(subject) for subject, _ in sorted_subjects[:5]]
    f.write(", ".join(top_5_subjects))
    f.write(") show particularly strong effects in:\n\n")
    
    # List the top metrics that show the strongest effects in the top subjects
    top_subject_data = quantum_df[quantum_df['subject'].isin([s for s, _ in sorted_subjects[:5]])]
    top_metrics = top_subject_data.groupby('metric')['difference'].mean().abs().sort_values(ascending=False).head(3)
    
    for metric, value in top_metrics.items():
        f.write(f"- **{metric}**: Average absolute difference = {value:.3f}\n")
    
    f.write("\n")
    
    f.write("## Visualizations\n\n")
    f.write("The following visualizations have been generated:\n\n")
    f.write("1. **band_power_differences.png**: Frequency band power differences between high and low ambiguity\n")
    f.write("2. **quantum_metrics_differences.png**: Quantum metrics differences between conditions\n")
    f.write("3. **correlation_matrix.png**: Correlations between all metrics\n")
    f.write("4. **quantum_neural_summary.png**: Summary visualization of key findings\n\n")
    
    f.write("## Future Directions\n\n")
    f.write("Future analyses could explore:\n\n")
    f.write("1. **Temporal dynamics** of quantum signatures during the transition from stimulus presentation to decision\n")
    f.write("2. **Cross-frequency coupling** as another potential quantum signature\n")
    f.write("3. **Source localization** to identify brain regions showing the strongest quantum signatures\n")
    f.write("4. **Comparison with behavioral data** to correlate quantum signatures with decision-making performance\n")
    f.write("5. **Comparison with other datasets** to test the generalizability of these findings\n")

print("\nAnalysis complete! Results saved to:", analysis_dir)
print("\nKey output files:")
print("1. key_findings.txt - Detailed statistical results and interpretations")
print("2. quantum_neural_report.md - Comprehensive report in Markdown format")
print("3. top_subjects.txt - Detailed analysis of subjects with strongest quantum signatures")
print("4. quantum_neural_summary.png - Summary visualization of key findings")
