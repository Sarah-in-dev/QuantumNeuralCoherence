#!/bin/bash
# Quick test script to verify analysis with a single subject

# Get the absolute path of the project root directory
# This assumes the script is running from the code/analysis directory
PROJECT_ROOT=$(cd ../../ && pwd)
echo "Project root: $PROJECT_ROOT"

# Set environment variables
export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH

# Create log directory if it doesn't exist
mkdir -p $PROJECT_ROOT/logs

# Variables - use absolute paths to be safer
DATA_DIR="${PROJECT_ROOT}/data/raw"
RESULTS_DIR="${PROJECT_ROOT}/results/test_single_subject"

# Look for the data file - check multiple potential locations
DATA_FILE=""
POTENTIAL_PATHS=(
  "${DATA_DIR}/EEG_data.mat"
  "${DATA_DIR}/necker_cube_eeg_Maksimenko/EEG_data.mat"
  "${PROJECT_ROOT}/data/raw/EEG_data.mat"
  "${PROJECT_ROOT}/EEG_data.mat"
)

for path in "${POTENTIAL_PATHS[@]}"; do
  if [ -f "$path" ]; then
    DATA_FILE="$path"
    echo "Found data file at: $DATA_FILE"
    break
  fi
done

# If still not found, ask user for path
if [ -z "$DATA_FILE" ]; then
  echo "Could not find EEG_data.mat in any of the expected locations."
  echo "Please enter the full path to the EEG_data.mat file:"
  read -p "> " USER_PATH
  
  if [ -f "$USER_PATH" ]; then
    DATA_FILE="$USER_PATH"
    echo "Using data file at: $DATA_FILE"
  else
    echo "Error: File not found at specified path: $USER_PATH"
    echo "Please download the dataset or check the file path."
    exit 1
  fi
fi

# Make sure the results directory exists
mkdir -p $RESULTS_DIR

echo "Starting test analysis with a single subject..."
echo "Data file: $DATA_FILE"
echo "Results will be saved to: $RESULTS_DIR"

# Run the analysis script with a single subject
python $PROJECT_ROOT/code/analysis/analyze_necker_eeg_group.py --data $DATA_FILE --output $RESULTS_DIR --subject 1 --skip-group

# Check if the analysis was successful
if [ $? -eq 0 ]; then
    echo "Test analysis completed successfully!"
    echo "Check the results in: $RESULTS_DIR"
    
    # List generated files
    echo "Generated files:"
    find $RESULTS_DIR -type f | sort
else
    echo "Test analysis failed. Check the error messages above."
fi
