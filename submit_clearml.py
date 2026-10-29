"""ClearML SDK submission of training job"""

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", default="muller")
    parser.add_argument("--task-name", default="poisson-scale-k1-pod")
    parser.add_argument("--max-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num-nodes", type=int, default=2)
    parser.add_argument("--tags", nargs="*", default=[])
    return parser.parse_args()


def main():
    args = parse_args()

    # Job config
    num_nodes = args.num_nodes
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
        task_name=args.task_name,
        task_type="training",
        repo="https://github.com/sparticlesteve/neuraloperators-TL-scaling.git",
        branch="clearml-testing",
        docker="nersc/pytorch:24.08.01",
        docker_bash_setup_script=container_setup_script,
        script="train.py",
        # Pin clearml: newer versions pull numpy 2.x, which breaks the image's torch (built for numpy 1.x)
        packages=["clearml==2.0.2"],
        #binary="/bin/bash",
        #script="./run_clearml.sh",
    )

    # Override some hyperparameters
    task.set_parameters({
        "Args/yaml_config": "./config/operators_poisson.yaml",
        "Args/config": "poisson-scale-k1_5",
        "Args/run_num": 1,
        #"Args/root_dir": "",
        "lr": args.lr,
        "max_epochs": args.max_epochs,
    })
    if args.tags:
        task.add_tags(args.tags)

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
        queue_name=args.queue,
    )

    print("\nTASK ENQUEUED:")
    pprint.pprint(enqueue_response)

if __name__ == "__main__":
    main()
