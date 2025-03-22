# Quantum Neural Coherence Analysis Report

## Analysis of Necker Cube EEG Dataset
*Generated on: 2025-03-22*

## Overview

This report analyzes EEG data from the Necker Cube experiment, focusing on quantum neural signatures during perceptual decision-making under varying ambiguity conditions. The analysis compares neural activity between high and low ambiguity conditions to identify potential quantum signatures in brain function.

## Key Findings

### Frequency Band Differences

| Band | High Ambiguity | Low Ambiguity | Difference | p-value | Significant |
|------|---------------|---------------|------------|---------|-------------|
| delta | 19.299 ± 2.389 | 20.104 ± 2.542 | -0.806 ± 1.430 | 0.0239 | ✓ |
| theta | 16.504 ± 2.773 | 16.783 ± 2.774 | -0.279 ± 1.031 | 0.2532 | ✗ |
| alpha | 16.476 ± 2.944 | 16.887 ± 2.831 | -0.411 ± 1.044 | 0.1022 | ✗ |
| beta | 13.001 ± 2.486 | 13.186 ± 2.636 | -0.185 ± 0.781 | 0.3145 | ✗ |
| gamma | -0.004 ± 3.488 | 0.108 ± 3.500 | -0.111 ± 1.147 | 0.6767 | ✗ |

### Quantum Neural Metrics

| Metric | High Ambiguity | Low Ambiguity | Difference | p-value | Significant |
|--------|---------------|---------------|------------|---------|-------------|
| Multiscale Entropy | 1.954 ± 0.165 | 1.939 ± 0.146 | 0.015 ± 0.156 | 0.6731 | ✗ |
| Phase Sync (Alpha) | 0.223 ± 0.144 | 0.217 ± 0.125 | 0.005 ± 0.175 | 0.8938 | ✗ |
| Phase Sync (Beta) | 0.198 ± 0.127 | 0.213 ± 0.133 | -0.016 ± 0.121 | 0.5834 | ✗ |
| TE (F→C) | 0.036 ± 0.002 | 0.034 ± 0.003 | 0.001 ± 0.004 | 0.1301 | ✗ |
| TE (C→F) | 0.034 ± 0.004 | 0.035 ± 0.004 | -0.001 ± 0.005 | 0.4577 | ✗ |
| TE Asymmetry | 0.001 ± 0.004 | -0.001 ± 0.004 | 0.002 ± 0.006 | 0.0927 | ✗ |

### Quantum Signatures Summary

### Interpretation

The analysis does not reveal significant quantum metrics, suggesting that the neural processing during this perceptual task is adequately explained by classical mechanisms.

### Individual Differences

There is considerable variation between subjects in quantum neural signatures. The 5 subjects with the strongest quantum signatures (Subjects 16, 20, 14, 18, 6) show particularly strong effects in:

- **Multiscale Entropy**: Average absolute difference = 0.098
- **Phase Sync (Beta)**: Average absolute difference = 0.033
- **Phase Sync (Alpha)**: Average absolute difference = 0.019

## Visualizations

The following visualizations have been generated:

1. **band_power_differences.png**: Frequency band power differences between high and low ambiguity
2. **quantum_metrics_differences.png**: Quantum metrics differences between conditions
3. **correlation_matrix.png**: Correlations between all metrics
4. **quantum_neural_summary.png**: Summary visualization of key findings

## Future Directions

Future analyses could explore:

1. **Temporal dynamics** of quantum signatures during the transition from stimulus presentation to decision
2. **Cross-frequency coupling** as another potential quantum signature
3. **Source localization** to identify brain regions showing the strongest quantum signatures
4. **Comparison with behavioral data** to correlate quantum signatures with decision-making performance
5. **Comparison with other datasets** to test the generalizability of these findings
