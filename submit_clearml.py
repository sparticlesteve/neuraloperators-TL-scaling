"""ClearML SDK submission of training job"""

import pprint
from typing import Dict, Any

import clearml
from clearml import Task


def get_task_summary(task: Task) -> Dict[str, Any]:
    summary = dict(
        id=task.id,
        task_type=task.task_type,
        project=task.get_project_name(),
        name=task.name,
        status=task.get_status(),
        tags=list(task.get_tags() or []),
        parameters=task.get_parameters_as_dict(),
        user_properties=task.get_user_properties(),
        script=task.get_script(),
    )
    return summary


def main():

    # Job config
    num_nodes = 2
    ntasks_per_node = 4
    cpus_per_task = 32
    container_setup_script = """
        export RANK=$SLURM_PROCID
        export LOCAL_RANK=$SLURM_LOCALID
        export WORLD_SIZE=$SLURM_NTASKS
    """

    # Starting with hardcoded configuration
    task = Task.create(
        project_name="FNO Tests",
        task_name="poisson-scale-k1-pod",
        task_type="training",
        repo="https://github.com/sparticlesteve/neuraloperators-TL-scaling.git",
        branch="clearml-testing",
        docker="nersc/pytorch:24.08.01",
        docker_bash_setup_script=container_setup_script,
        script="train.py",
        #binary="/bin/bash",
        #script="./run_clearml.sh",
    )

    # Override some hyperparameters
    task.set_parameters({
        "Args/yaml_config": "./config/operators_poisson.yaml",
        "Args/config": "poisson-scale-k1_5",
        "Args/run_num": 13,
        #"Args/root_dir": "",
        "lr": 0.001,
        "max_epochs": 4,
    })

    # SLURM job settings
    task.set_user_properties(
        num_nodes=num_nodes,
        ntasks_per_node=ntasks_per_node,
        cpus_per_task=cpus_per_task,
    )

    # Print the configuration
    print("\nCREATED NEW TASK:")
    pprint.pprint(get_task_summary(task))

    # Enqueue the task
    enqueue_response = Task.enqueue(
        task=task,
        queue_name="experimental",
    )

    print("\nTASK ENQUEUED:")
    pprint.pprint(enqueue_response)

if __name__ == "__main__":
    main()
