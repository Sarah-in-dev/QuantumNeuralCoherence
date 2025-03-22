# create a file called explore_data.py in your code directory
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os

# Path to the data file
data_path = '/blue/mcb4324/share/sdavidson2/QuantumNeuralCoherence/data/raw/EEG_data.mat'

# Load the data
data = scipy.io.loadmat(data_path)

# Print the keys to see the structure of the data
print("Keys in the MATLAB file:")
print(data.keys())

# Print information about one subject's data
if 'data' in data:
    subject_data = data['data'][0]  # First subject
    print(f"\nNumber of subjects: {len(data['data'])}")
    
    # Assuming structure based on the description you provided
    if hasattr(subject_data, 'Trial'):
        print(f"Number of trials: {len(subject_data.Trial)}")
        print(f"Number of channels: {subject_data.Trial[0].shape[0]}")
        print(f"Trial length (samples): {subject_data.Trial[0].shape[1]}")
    
    # Print other available fields
    print("\nAvailable fields for each subject:")
    for field in dir(subject_data):
        if not field.startswith('_'):
            print(f"- {field}")
