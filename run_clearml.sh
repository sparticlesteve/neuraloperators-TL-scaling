#!/bin/bash

# Dump entire env for testing
env | grep CLEARML

set -ex

# Container configuration
SHIFTER_IMAGE=nersc/pytorch:24.08.01
SHIFTER_MODULES=gpu,nccl-plugin

# Distributed training configuration
export MASTER_ADDR=$(scontrol show hostnames | head -n 1)
export MASTER_PORT=29507

# Training configuration
config_file=./config/operators_poisson.yaml
config="poisson-scale-k1_5"
run_num=$SLURM_JOB_ID
results_dir=$SCRATCH/clearml_tests/results

mkdir -p ${results_dir}

# Install clearml package in local directory
export PYTHONUSERBASE=".local/clearml_testing_fno"
shifter --image=$SHIFTER_IMAGE --module=$SHIFTER_MODULES \
    bash -c "pip install clearml"

launch_cmd="torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc-per-node=${SLURM_GPUS_PER_TASK:-4} --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT" 
script_cmd="train.py --yaml_config=$config_file --config=$config --run_num=$run_num --root_dir=$results_dir"

shifter --image=$SHIFTER_IMAGE --module=$SHIFTER_MODULES \
    bash -c "$launch_cmd $cmd"
