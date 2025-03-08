#!/bin/bash

# This script will use the clearml data CLI to upload the FNO dataset

DATA_DIR=${DATA_DIR:-$SCRATCH/clearml_tests/poisson}
PROJECT_NAME=${PROJECT_NAME:-"FNO Tests"}
DATASET_NAME=${DATASET_NAME:-"FNO Poisson Mini Dataset"}

# Software setup
module use /global/common/software/nersc/pe/modulefiles/latest
module load pytorch/2.6.0
pip list 2>/dev/null | grep -q clearml || pip install clearml

# Ensure we are logged in to our specific ClearML instance interactively
if [ ! -f ~/clearml.conf ]; then
    echo "ClearML configuration file not found. Running clearml-init..."
    clearml-init
fi

set -x

# Create FNO testing dataset in ClearML and upload the DATA_DIR
clearml-data create --project "$PROJECT_NAME" --name "$DATASET_NAME"
clearml-data add --files "$DATA_DIR"
clearml-data list
clearml-data close
