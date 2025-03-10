#!/bin/bash
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 30
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --image=nersc/pytorch:24.08.01
#SBATCH --module=gpu,nccl-plugin

export MASTER_ADDR=$(hostname)
config_file=./config/operators_poisson.yaml
config="poisson-scale-k1_5"
run_num="00"

# path/to/logs
results_dir=$SCRATCH/clearml_tests/results
mkdir -p ${results_dir}

# Install clearml package
export PYTHONUSERBASE=".local/clearml_testing_fno"
shifter bash -c "pip install clearml"

cmd="python train.py --yaml_config=$config_file --config=$config --run_num=$run_num --root_dir=$results_dir"
srun -l shifter bash -c "
    source export_DDP_vars.sh
    $cmd
"

# if wandb sweeps
#sweep_id="e8me2vut"
#cmd_sweep="python train.py --yaml_config=$config_file --config=$config --run_num=$run_num --root_dir=$results_dir --sweep_id=$sweep_id"
#srun -u --mpi=pmi2 --nodes=1 --ntasks-per-node=4 --cpus-per-task=32 --gpus-per-node=4 shifter --module gpu --image=${image} \
#    bash -c "
#    source export_DDP_vars.sh
#    $cmd_sweep
#    "
