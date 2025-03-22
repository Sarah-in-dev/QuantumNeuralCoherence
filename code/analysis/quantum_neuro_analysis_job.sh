#!/bin/bash
#SBATCH --job-name=quantum_neuro
#SBATCH --output=logs/quantum_neuro_%j.out
#SBATCH --error=logs/quantum_neuro_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sdavidson2@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=04:00:00
#SBATCH --partition=hpg-default
#SBATCH --account=mcb4324

# Print execution info
echo "Job running on $(hostname)"
echo "Job started at $(date)"
echo "Directory: $(pwd)"

# Create log directory if it doesn't exist
mkdir -p logs

# Load required modules
module load conda

# Activate environment
echo "Activating quantum_neuro conda environment"
conda activate quantum_neuro

# Project directory (change this to your project root)
PROJECT_DIR="/blue/mcb4324/share/sdavidson2/QuantumNeuralCoherence"
cd $PROJECT_DIR

# Make sure project path is added to PYTHONPATH
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

# Make sure data directory exists
DATA_DIR="$PROJECT_DIR/data/raw"
RESULTS_DIR="$PROJECT_DIR/results"
mkdir -p $RESULTS_DIR

# Look for the data file in several potential locations
DATA_FILE=""
POTENTIAL_PATHS=(
  "${DATA_DIR}/EEG_data.mat"
  "${DATA_DIR}/necker_cube_eeg_Maksimenko/EEG_data.mat"
  "$PROJECT_DIR/EEG_data.mat"
)

for path in "${POTENTIAL_PATHS[@]}"; do
  if [ -f "$path" ]; then
    DATA_FILE="$path"
    echo "Found data file at: $DATA_FILE"
    break
  fi
done

# Check if the data file was found
if [ -z "$DATA_FILE" ]; then
    echo "Error: EEG_data.mat not found in any expected location."
    echo "Searched in:"
    for path in "${POTENTIAL_PATHS[@]}"; do
      echo "  - $path"
    done
    echo "Please download the dataset and place it in one of these locations."
    exit 1
fi

# Run the analysis script
echo "Starting Necker Cube EEG analysis"

# Option 1: Analyze all subjects
python code/analysis/analyze_necker_eeg_group.py --data $DATA_FILE --output $RESULTS_DIR

# Option 2: Analyze a specific subject
# python code/analysis/analyze_necker_eeg_group.py --data $DATA_FILE --output $RESULTS_DIR --subject 1

# Option 3: Analyze a range of subjects
# python code/analysis/analyze_necker_eeg_group.py --data $DATA_FILE --output $RESULTS_DIR --start 0 --end 9

# Print job completion information
echo "Job completed at $(date)"
